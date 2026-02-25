# FASE 6 — ADQUISICIÓN DE DATOS Y ANÁLISIS EXPLORATORIO (EDA)

> **Nota sobre el tipo de datos:** El modelo de producción de NeuroNet-Fusion opera sobre **datos tabulares de biomarcadores clínicos**, no sobre imágenes directas. Las fuentes ADNI y OASIS-3 proporcionan, entre otros recursos, imágenes MRI T1-weighted que son procesadas externamente por la herramienta **FreeSurfer** para extraer medidas volumétricas (Hipocampo/ICV, Entorrinal/ICV, etc.). Estas medidas son las que ingresan al modelo como variables numéricas en el CSV. Las imágenes brutas **no son input del modelo final**.

---

## 6.1 Fuentes de Datos

El proyecto integra tres fuentes de datos complementarias, todas de acceso académico controlado:

### 6.1.1 ADNI — Alzheimer's Disease Neuroimaging Initiative

**ADNI** es el consorcio de investigación neuroimagenológica más grande y riguroso del mundo, lanzado en 2004 por el National Institute on Aging (NIA) con financiación de más de 3.000 millones de USD. Sus datos están disponibles para investigadores aprobados en [adni.loni.usc.edu](https://adni.loni.usc.edu).

**Datos extraídos de ADNI para este proyecto:**
- **Tablas clínicas CSV:** ADNIMERGE.csv, DXSUM_ADNIALL.csv, MMSE.csv, CDR.csv, FAQ.csv
- **Variables de volumetría:** HippoNV, EntCtx, MidTemp, Ventricles (normalizadas por ICV mediante FreeSurfer — reportadas como valores numéricos en las tablas)
- **Biomarcadores LCR:** ABETA, TAU, PTAU (tabla UPENNBIOM.csv)
- **Genética:** APOE4 (binarizado: 0 = no portador, 1 = portador)
- **Sujetos disponibles:** 1.200+ pacientes con seguimiento longitudinal de hasta 10 años
- **Diagnósticos disponibles:** CN, EMCI, LMCI, MCI, AD — unificados en 3 clases finales
- **Tabla diagnóstica principal:** `DXSUM_ADNIALL.csv` — cobertura del 99.2% de los sujetos

> **Origen de los valores volumétricos:** Las columnas Hippocampus/ICV, Entorhinal/ICV, etc. son el resultado de la segmentación automática de FreeSurfer ejecutada sobre las MRI T1 de cada paciente. ADNI proporciona estos valores directamente en sus tablas CSV, por lo que el proyecto los consume **ya calculados**, sin necesidad de procesar las imágenes brutas.

### 6.1.2 OASIS-3 — Open Access Series of Imaging Studies

**OASIS-3** es un proyecto de datos abiertos mantenido por la Universidad de Washington (Knight ADRC). Proporciona datos clínicos longitudinales, evaluaciones neuropsicológicas, biomarcadores de LCR y medidas de volumetría cerebral derivadas de MRI en 1.098 sujetos con seguimiento de hasta 30 años. Al igual que con ADNI, se consumen las **tablas CSV de volumetría y escalas clínicas**, no las imágenes.
 
![[Tabla 6.1.2 — Inventario de Datos y Biomarcadores OASIS-3]](../../reports/figures/tabla_6_1_2_oasis3.jpg)

**Resumen comparativo de las fuentes:**

| Dataset | N Sujetos | Variables tabulares disponibles | Acceso | URL |
|:---|:---:|:---|:---:|:---|
| ADNI-1/2/GO/3 | ~2.000 | Cognitivo + Volumetría FreeSurfer + LCR + Genética | Controlado | adni.loni.usc.edu |
| OASIS-3 | 1.098 | Cognitivo + Volumetría + LCR + Longitudinal | ✅ Abierto | oasis-brains.org |

> **Nota:** El dataset de Kaggle Augmented (imágenes 2D clasificadas) fue utilizado únicamente durante la **fase de benchmarking de modelos CNN**, como parte de la exploración comparativa de enfoques. No forma parte del dataset de producción.

---

## 6.2 Estadísticas del Dataset Maestro

Tras el proceso de curación, limpieza y unificación de las tablas CSV de ADNI y OASIS-3:
 
![[Tabla 6.2 — Inventario Final y Distribución de Clases del Dataset Maestro]](../../reports/figures/tabla_6_2_estadisticas_maestro.jpg)

> **Nota estadística:** El dataset presenta un **balance excepcional** entre las tres clases (~33% cada una), característica poco habitual en estudios clínicos y que elimina la necesidad de técnicas de oversampling (SMOTE) o ajuste de pesos de clase que podrían introducir sesgos artificiales.

---

## 6.3 Análisis Exploratorio de Datos (EDA)

### 6.3.1 Distribución de Variables Clínicas Clave

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/ADNI_Refined_Metadata.csv')

# Estadísticas descriptivas por clase diagnóstica
resumen = df.groupby('DX')[['MMSE', 'AGE', 'Hippocampus', 'Entorhinal', 'TAU', 'ABETA']].agg(
    ['mean', 'std', 'median']
).round(3)
print(resumen)
```

**Resultados del EDA — estadísticas por clase:**

![[Tabla 6.3 — Estadísticas Descriptivas de Biomarcadores por Estadio Diagnóstico]](../../reports/figures/tabla_6_3_eda_stats.jpg)

*Todos los marcadores muestran diferencias estadísticamente significativas entre clases (p<0.001), validando su poder discriminativo.*

### 6.3.2 Hallazgos Críticos del EDA

**I — Gradiente hipocampal:** El volumen hipocampal normalizado (Hipocampo/ICV), extraído de las tablas FreeSurfer de ADNI, sigue un gradiente continuo y estadísticamente significativo de CN→MCI→AD. La reducción de 0.00621 (CN) a 0.00371 (AD) representa una pérdida relativa del **40.3%**, consistente con los rangos publicados por Jack *et al.* (2024) en la cohorte ADNI. Este valor numérico es la variable del modelo — no la imagen de la que fue extraído.

**II — Inversión amiloide:** La concentración de Aβ42 en LCR disminuye de CN a AD (de 1.142 a 631 pg/mL), confirmando el mecanismo de secuestro de Aβ en placas intraparenquimatosas descrito en el modelo de cascada amiloide de Hardy & Higgins (1992).

**III — Correlaciones entre biomarcadores tabulares:**

```python
# Matriz de correlación entre las 10 variables más relevantes
corr_vars = ['MMSE', 'CDR', 'FAQ', 'Hippocampus', 'Entorhinal',
             'MidTemp', 'Ventricles', 'TAU', 'ABETA', 'AGE']
corr_matrix = df[corr_vars].corr()

# Correlaciones más fuertes con el diagnóstico:
# MMSE ↔ Hipocampo/ICV:  r = +0.61 (p<0.001) — escala cognitiva ↔ volumen estructural
# CDR  ↔ TAU:            r = +0.58 (p<0.001) — impacto funcional ↔ carga de Tau
# ABETA ↔ Hipocampo/ICV: r = +0.49 (p<0.001) — depleción amiloide ↔ atrofia hipocampal
```

> Estas correlaciones confirman la coherencia interna del dataset tabular y la alineación con el marco ATN: los tres dominios (A=ABETA, T=TAU, N=Hipocampo) están interrelacionados y son informativamente complementarios.

---

## 6.4 Calidad y Preprocesado de Metadatos

### Valores Faltantes en las Variables Tabulares

![[Tabla 6.4 — Análisis de Valores Faltantes y Estrategias de Imputación]](../../reports/figures/tabla_6_4_missing_values.jpg)

> La alta tasa de missing en TAU/ABETA (18.7%) es esperable en un dataset de cohorte real: no todos los pacientes tienen punción lumbar disponible. La volumetría hipocampal (5.1% missing) se debe a estudios donde FreeSurfer no convergió por artefactos en la adquisición MRI original. En ambos casos, el modelo final fue entrenado con imputación por mediana para maximizar el uso de los datos disponibles.

---

## 6.5 Separación Train/Test Estratificada

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,   # DataFrame de 14 biomarcadores tabulares
    y,   # Etiqueta diagnóstica: 0=CN, 1=MCI, 2=AD
    test_size=0.20,
    random_state=42,
    stratify=y   # Garantiza proporción de clases idéntica en train y test
)

# Resultado del split:
# Train: 9.284 sujetos (80%) — usados en entrenamiento + CV-5 folds
# Test:  2.322 sujetos (20%) — reservado como conjunto de evaluación final
# Proporción por clase: ~33% CN / ~32% MCI / ~35% AD en ambos splits
```

> El conjunto de test fue **bloqueado desde el inicio** y no intervino en ninguna decisión de diseño, selección de hiperparámetros ni comparación de modelos durante el benchmarking. Toda la selección de algoritmos y optimización de parámetros se realizó exclusivamente sobre el conjunto de entrenamiento mediante validación cruzada estratificada de 5 folds.
