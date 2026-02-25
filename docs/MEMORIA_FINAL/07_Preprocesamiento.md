# FASE 7 — PREPROCESAMIENTO Y NORMALIZACIÓN

> **Nota importante sobre el alcance:** Este capítulo documenta **dos pipelines de preprocesamiento** desarrollados durante el proyecto:
> 1. **Pipeline de Imagen MRI (Fase de Investigación):** Desarrollado para el benchmarking comparativo de modelos CNN sobre volúmenes 3D. Los resultados de este benchmarking (60% accuracy) motivaron la decisión de adoptar el enfoque tabular en el modelo final.
> 2. **Pipeline de Biomarcadores Tabulares (Modelo de Producción):** El preprocesado que alimenta el modelo XGBoost final. Los valores de volumetría (Hipocampo/ICV, etc.) son resultados numéricos de la segmentación FreeSurfer, no imágenes.

---

## 7.1 Pipeline de Preprocesamiento de Imagen MRI (3D)

El preprocesamiento de neuroimagen es la fase más crítica del pipeline: la calidad del entrenamiento depende directamente de la calidad de los volúmenes de entrada. Se implementó un pipeline en 5 pasos progresivos:

### Paso 1 — Conversión DICOM → NIfTI

```python
import dicom2nifti
import os

def convert_series_to_nifti(dicom_dir: str, output_path: str):
    """
    Convierte una serie DICOM (160-200 cortes T1) a un volumen NIfTI unificado.
    Requiere que los cortes estén en una carpeta, ordenados por InstanceNumber.
    """
    try:
        dicom2nifti.convert_directory(
            dicom_directory=dicom_dir,
            output_folder=output_path,
            compression=True,       # Genera .nii.gz
            reorient=True           # Reorienta a espacio RAS estándar
        )
        print(f"[OK] Convertido: {os.path.basename(dicom_dir)}")
    except Exception as e:
        print(f"[ERROR] {dicom_dir}: {e}")
```

![[Código 7.1 Paso 1 — Conversión DICOM a NIfTI]](../../reports/figures/codigo_7_paso1_dicom_nifti.png)

**Resultado:** 135 volúmenes T1-weighted convertidos correctamente (tasa de éxito 100%).  
**Tamaño medio por volumen NIfTI:** ~280 MB (sin comprimir) → ~45 MB (.nii.gz).

### Paso 2 — Reorientación Anatómica RAS

Todos los volúmenes fueron reorientados al espacio **RAS (Right-Anterior-Superior)**, asegurando que el eje X apunte a la derecha, el Y hacia anterior y el Z superior. Esto es crítico para la consistencia anatómica entre escáneres de diferentes fabricantes y magnitudes de campo (1.5T vs 3T).

```python
import nibabel as nib
import numpy as np

def reorient_to_ras(nifti_path: str) -> nib.Nifti1Image:
    """Reorienta el volumen a espacio RAS estándar."""
    img    = nib.load(nifti_path)
    ras_img = nib.as_closest_canonical(img)
    return ras_img
```

![[Código 7.1 Paso 2 — Reorientación Anatómica RAS]](../../reports/figures/codigo_7_paso2_reorientacion_ras.png)

### Paso 3 — Normalización de Intensidad Z-Score

La variabilidad entre escáneres 1.5T y 3.0T produce diferencias en las intensidades de los voxels de hasta un 300%. La normalización Z-score por volumen elimina este sesgo:

$$z_{voxel} = \frac{x_{voxel} - \mu_{brain}}{\sigma_{brain}}$$

donde $\mu_{brain}$ y $\sigma_{brain}$ se calculan **solo sobre los voxels con señal** (valor > 0), excluyendo el fondo de aire.

```python
def normalize_zscore(volume: np.ndarray) -> np.ndarray:
    """
    Normalización Z-score sobre voxels con señal.
    Resultado: media=0, std=1 en tejido cerebral.
    """
    mask   = volume > 0
    mean_v = volume[mask].mean()
    std_v  = volume[mask].std()
    
    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[mask] = (volume[mask] - mean_v) / (std_v + 1e-8)
    return normalized
```

![[Código 7.1 Paso 3 — Normalización de Intensidad Z-Score]](../../reports/figures/codigo_7_paso3_normalizacion_zscore.png)

### Paso 4 — Segmentación de Bounding Box

Para eliminar el espacio vacío (aire) alrededor del cráneo y centrar la red en el tejido cerebral:

```python
def crop_to_brain(volume: np.ndarray) -> np.ndarray:
    """Recorta el volumen al bounding box del cerebro."""
    nonzero = np.argwhere(volume > 0)
    if len(nonzero) == 0:
        return volume
    min_coords = nonzero.min(axis=0)
    max_coords = nonzero.max(axis=0) + 1
    return volume[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
    ]
```

![[Código 7.1 Paso 4 — Segmentación de Bounding Box]](../../reports/figures/codigo_7_paso4_bounding_box.png)

### Paso 5 — Redimensionado Uniforme (128×128×128)

```python
from scipy.ndimage import zoom

def resize_volume(volume: np.ndarray, target=(128, 128, 128)) -> np.ndarray:
    """Redimensiona el volumen cerebral a dimensiones estándar."""
    factors = [t/s for t, s in zip(target, volume.shape)]
    return zoom(volume, factors, order=1)  # Interpolación bilineal
```

![[Código 7.1 Paso 5 — Redimensionado Uniforme 128x128x128]](../../reports/figures/codigo_7_paso5_resize_128.png)

---

## 7.2 Pipeline de Preprocesamiento de Datos Tabulares

### 7.2.1 Inventario de Variables

![[Tabla 7.2.1 — Inventario de Biomarcadores y Rangos Clínicos]](../../reports/figures/tabla_7_2_1_inventario.jpg)

### 7.2.2 Tratamiento de Valores Faltantes

La integridad de los datos clínicos es uno de los mayores desafíos en neurociencia. En NeuroNet-Fusion, se ha evitado la eliminación de registros (*listwise deletion*) para no perder el 20% de la cohorte que carecía de punciones lumbares. En su lugar, se ha implementado una **Estrategia de Imputación Estratificada por Clase**.

#### Lógica de Imputación:

```python
from sklearn.impute import SimpleImputer
import pandas as pd

# Estrategia: imputación por mediana, preservando la señal de clase
def impute_by_class(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Imputa valores faltantes utilizando la mediana de cada grupo diagnóstico.
    Esto evita que un paciente AD reciba valores 'normales' (CN) que 
    difuminarían la frontera de decisión del modelo.
    """
    for cls in df[target_col].unique():
        mask = df[target_col] == cls
        # Calculamos la mediana específica para este estadio (CN, MCI o AD)
        medians = df[mask].median(numeric_only=True)
        df.loc[mask] = df.loc[mask].fillna(medians)
    
    # Manejo de biomarcadores moleculares (Informative Missingness)
    # Creamos un flag binario para indicar si el dato de LCR era original o imputado
    df['biomarker_available'] = df['ABETA'].notna().astype(int)
    
    return df
```

![[Código 7.2.2 — Estrategia de Imputación por Clase e Inclusión de Flags de Ausencia]](../../reports/figures/codigo_7_2_2_imputacion.png)

**Justificación y Resultados:**
1.  **Preservación de la Señal:** Al imputar por mediana de clase, nos aseguramos de que el modelo aprenda las distribuciones típicas de cada estadio. Imputar con la mediana global "limpiaría" artificialmente los datos, ocultando la patología.
2.  **Ausencia Informativa:** Dado que en la práctica clínica la punción lumbar (TAU/ABETA) se realiza con mayor frecuencia en casos de sospecha alta, la ausencia de este dato es en sí misma informativa. El modelo XGBoost puede utilizar el flag `biomarker_available` para ajustar su incertidumbre.
3.  **Impacto:** Se logró recuperar el **100% de los 11.606 registros**, maximizando el poder estadístico del entrenamiento.

### 7.2.3 Escalado y Codificación

Una de las contribuciones críticas de este proyecto es la **Normalización por Volumen Intracraneal (ICV)**. Las medidas volumétricas cerebrales (como el tamaño del Hipocampo) están correlacionadas con el tamaño total del cráneo del individuo. Sin normalización, un hipocampo sano en una persona pequeña podría confundirse con un hipocampo atrofiado en una persona grande.

El proceso asegura que el modelo aprenda **proporciones de atrofia**, no dimensiones absolutas:

$$V_{norm} = \frac{V_{absoluto}}{ICV}$$

#### Implementación del Pipeline de Escalado:

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def preprocess_tabular_features(X_train, y_train):
    # 1. Normalización Biológica (ICV)
    # Se realiza sobre las columnas de volumetría bruta de FreeSurfer
    vol_cols = ['Hippocampus', 'Entorhinal', 'MidTemporal', 'Ventricles']
    for col in vol_cols:
        X_train[f'{col}_Norm'] = X_train[col] / X_train['ICV']
    
    # 2. Escalado Estadístico (Z-Score)
    # Relevante para algoritmos sensibles a la magnitud (MLP, SVM) y 
    # para la estabilidad de convergencia en Gradient Boosting.
    continuous_cols = [
        'BCMMSE', 'BCFAQ', 'entry_age', 'PTEDUCAT', 
        'Hippo_Norm', 'Ento_Norm', 'MidTemp_Norm', 'Vent_Norm',
        'ABETA', 'TAU', 'PTAU'
    ]
    scaler = StandardScaler()
    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    
    # 3. Codificación de Etiquetas (Target)
    # Conversión de categorías clínicas a índices numéricos
    le = LabelEncoder()
    # Mapeo: CN -> 0, MCI -> 1, AD -> 2
    y_encoded = le.fit_transform(y_train)
    
    # Guardar objetos para inferencia en producción
    import joblib
    joblib.dump(scaler, 'models/scaler_production.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    
    return X_train, y_encoded, le
```

![[Código 7.2.3 — Pipeline de Escalado, Normalización ICV y Codificación]](../../reports/figures/codigo_7_2_3_escalado_codificacion.png)

**Explicación Técnica:**
- **StandardScaler:** Transforma las variables para que tengan media 0 y desviación estándar 1. Esto es vital para que biomarcadores con escalas muy diferentes (ej. MMSE de 0-30 vs. ABETA de 200-2000) tengan el mismo peso inicial en el modelo.
- **LabelEncoder:** Transforma las categorías diagnósticas en un formato vectorial procesable por la función de pérdida `multi:softprob` de XGBoost.
- **Persistencia:** Tanto el `scaler` como el `le` se guardan físicamente para asegurar que los datos introducidos por el médico en la aplicación Streamlit reciban exactamente la misma transformación que los datos de entrenamiento.

---

## 7.3 Data Augmentation para MRI 2D

Para el entrenamiento del modelo CNN sobre cortes 2D (dataset Kaggle), se implementaron transformaciones clínicamente conservadoras:

```python
from torchvision import transforms

def get_train_transforms(img_size=(224, 224)):
    """
    Augmentation conservador para imágenes MRI:
    - Rotación limitada (±15°): simula variación de posición del paciente
    - Flip horizontal: el cerebro es casi simétrico
    - Color jitter mínimo: simula variaciones de contraste inter-escáner
    - NO flip vertical: anatómicamente incorrecto
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]    # Estadísticas ImageNet
        )
    ])

def get_val_transforms(img_size=(224, 224)):
    """Sin augmentation para validación/test."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

![[Código 7.3 — Pipeline de Data Augmentation para Imágenes MRI 2D (Benchmarking)]](../../reports/figures/codigo_7_3_data_augmentation.png)
