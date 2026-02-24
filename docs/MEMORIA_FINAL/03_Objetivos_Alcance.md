# FASE 3 — OBJETIVOS Y ALCANCE

---

## 3.1 Objetivo General

Desarrollar, entrenar y desplegar un sistema de Inteligencia Artificial basado en **biomarcadores clínicos y moleculares** (**NeuroNet-Fusion**) capaz de clasificar automáticamente el estadio de la enfermedad de Alzheimer en tres categorías diagnósticas — Cognitivamente Normal (CN), Deterioro Cognitivo Leve (MCI) y Alzheimer Establecido (AD) — integrando 14 variables multidominio (neuropsicológicas, volumétricas normalizadas y moleculares del LCR), alcanzando una precisión ≥85% con interpretabilidad clínica completa alineada con el marco ATN-NIA-AA 2018.

> **Nota metodológica:** Aunque durante las fases de investigación se exploró el procesamiento de imágenes MRI (pipeline DICOM→NIfTI y benchmarking de CNNs), el benchmarking empírico demostró que los **biomarcadores tabulares** extraídos de la volumetría cerebral normalizada y del LCR — cuando se combinan con escalas neuropsicológicas — superan a los modelos de imagen bruta en precisión y eficiencia computacional. El **modelo de producción final opera exclusivamente sobre datos tabulares**.

---

## 3.2 Objetivos Específicos

| ID | Objetivo Específico | Herramienta / Método | Resultado Esperado |
|:---:|:---|:---|:---|
| OE-01 | Adquirir y unificar datos de ADNI y OASIS-3 | Pandas, scripts Python | Dataset maestro de 11.606 sujetos |
| OE-02 | Explorar el procesamiento volumétrico MRI como fase de investigación | `dicom2nifti`, nibabel | 135 volúmenes 3D procesados (benchmarking) |
| OE-03 | Realizar benchmarking comparativo de 12 familias de algoritmos ML | Scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch | Selección empírica del algoritmo óptimo |
| OE-04 | Diseñar y optimizar el modelo de clasificación sobre biomarcadores | XGBoost + Optuna | Accuracy ≥86% sobre 14 biomarkers |
| OE-05 | Entrenar modelo con búsqueda automática de hiperparámetros | Optuna (100 trials, CV-5) | Convergencia estable; AUC ≥0.89 |
| OE-06 | Validar con métricas clínicamente relevantes | Recall, F1, AUC-ROC, Kappa | Sensibilidad ≥90% en AD moderado |
| OE-07 | Analizar importancia de biomarcadores (SHAP) | `shap` library (TreeSHAP) | Ranking clínico de los 5 factores determinantes |
| OE-08 | Desplegar sistema web de soporte clínico interactivo | Streamlit | App CDSS con informe ATN descargable |

![[Tabla 3.2 — Objetivos Específicos del Proyecto NeuroNet-Fusion]](../../reports/figures/tabla_3_2_objetivos.jpg)

---

## 3.3 Requerimientos Funcionales

### RF-01: Ingesta de Datos Clínicos
- El sistema debe soportar la carga de archivos CSV con las 14 variables biomarcadoras.
- Debe unificar pacientes de ADNI y OASIS-3 bajo un identificador único (RID/Subject_ID).
- Los datos volumétricos (Hipocampo/ICV, Entorrinal/ICV, etc.) son el resultado de la segmentación FreeSurfer, **no imágenes brutas**: se incorporan como variables numéricas en el CSV.

### RF-02: Preprocesamiento de Biomarcadores Tabulares
- Imputación de valores faltantes por mediana estratificada por clase diagnóstica.
- Escalado estándar (`StandardScaler`) sobre variables continuas.
- Codificación binaria de variables categóricas (APOE4: 0/1).
- Generación de flag binario `biomarker_available` para las variables con >10% missing (TAU, ABETA).

### RF-03: Benchmarking y Selección de Modelo
- Evaluación comparativa de ≥10 familias de algoritmos (ML clásico, ensemble, Deep Learning tabular, CNN sobre imagen).
- Validación cruzada estratificada K-Fold (k=5) con `random_state=42`.
- Métricas reportadas: Accuracy, F1-Score Weighted, AUC-ROC, tiempo de entrenamiento.

### RF-04: Explicabilidad e Interpretabilidad del Modelo Tabular
- Análisis SHAP (beeswarm, waterfall y force plots) sobre el modelo XGBoost.
- Ranking de importancia de los 14 biomarcadores por ganancia (Gain) y SHAP value.
- Narrativa clínica automática generada a partir de los valores SHAP más significativos.

### RF-05: Sistema de Soporte a la Decisión Clínica (CDSS)
- Interfaz web con entrada manual de los 14 biomarcadores mediante sliders y campos numéricos.
- Motor de inferencia en tiempo real con probabilidades por clase y perfil ATN calculado.
- Generación de dictamen neurológico narrativo estructurado descargable en formato `.txt`.

---

## 3.4 Requerimientos No Funcionales

| ID | Requerimiento | Criterio de Aceptación |
|:---:|:---|:---|
| RNF-01 | **Reproducibilidad** | `requirements.txt` + `random_state=42` en todos los modelos |
| RNF-02 | **Portabilidad** | Compatible con Windows y Linux; dependencias solo de Python estándar |
| RNF-03 | **Modularidad** | Código en módulos: `data/`, `models/`, `src/`, `reports/` |
| RNF-04 | **Eficiencia en inferencia** | Predicción en <100ms por paciente (XGBoost tabular sin GPU en producción) |
| RNF-05 | **Usabilidad clínica** | Interfaz comprensible para personal sanitario sin conocimientos de IA |
| RNF-06 | **Seguridad Térmica** | Aplicación de ThermalThrottleCallback para protección de CPU/GPU |

![[Tabla 3.4 — Requerimientos No Funcionales del Sistema NeuroNet-Fusion]](../../reports/figures/tabla_3_4_rnf.jpg)

---

## 3.5 Alcance y Limitaciones Declaradas

**Dentro del alcance:**
- Clasificación sobre 14 biomarcadores tabulares: CN, MCI, AD.
- Datos de pacientes adultos ≥50 años de las cohortes ADNI y OASIS-3.
- Validación en split de test independiente del 20% (estratificado).
- Aplicación web interactiva con informe ATN descargable.
- Exploración comparativa (benchmarking) del enfoque por imagen vs. biomarcadores.

**Fuera del alcance del modelo de producción:**
- Análisis directo de imágenes MRI como input del modelo final (se usa la volumetría normalizada derivada de FreeSurfer, no la imagen bruta).
- Datos longitudinales de seguimiento (análisis de progresión temporal).
- Integración de datos genómicos de secuenciación completa (solo APOE4 binarizado).
- Validación prospectiva en entorno hospitalario real (requiere aprobación ética del CE).
- Diagnóstico diferencial con otras demencias (Lewy Body, DFT, Demencia Vascular).

---

## 3.6 Pila Tecnológica

```python
# Core del Entorno
Python        == 3.12
joblib        >= 1.3      # Serialización y carga del modelo

# Machine Learning — Modelo de Producción
xgboost       >= 2.0      # Modelo champion
lightgbm      >= 4.0      # Modelo ensemble (votación blanda)
catboost      >= 1.2      # Modelo ensemble (votación blanda)
scikit-learn  >= 1.3      # Preprocesado, métricas, validación cruzada
shap          >= 0.44     # Explicabilidad TabularSHAP / TreeSHAP

# Machine Learning — Benchmarking (fase de investigación)
# PyTorch, torchvision, nibabel, dicom2nifti → usados en la
# fase experimental de CNNs y procesamiento volumétrico.
# NO son dependencias del modelo de producción.

# Aplicación Web CDSS
streamlit     >= 1.30
plotly        >= 5.18
pandas        >= 2.1

# Análisis y Documentación
numpy         >= 1.26
matplotlib    >= 3.8
seaborn       >= 0.13
```

![[Tabla 3.6 — Pila Tecnológica Core del Sistema NeuroNet-Fusion]](../../reports/figures/tabla_3_6_pila_tecnologica.jpg)
