# Fase 4: Adquisición de Datos y Análisis Exploratorio (EDA)

## 4.1 Fuentes de Información: El Dataset ADNI
El proyecto utiliza el ecosistema de **ADNI (Alzheimer's Disease Neuroimaging Initiative)** como fuente principal. Tras un proceso de curación profunda, se ha consolidado un registro maestro de alta fidelidad:

- **Estrategia de Etiquetado Masivo**: Mediante la integración de la tabla primaria `DXSUM_ADNIALL.csv`, se logró una cobertura diagnóstica del **99.2%** sobre los sujetos detectados físicamente.
- **Inventario Refinado (MRI T1-weighted)**: Se han seleccionado **11,606 imágenes representativas** (una por visita), filtrando duplicados y capturas de baja resolución (Localizers).
- **Consistencia Multimodal**: Cada imagen está vinculada unívocamente a su diagnóstico clínico y biomarcadores cognitivos (MMSE, ADAS13) en el archivo `ADNI_Refined_Metadata.csv`.

## 4.2 Análisis Exploratorio (EDA)
### Hallazgos en el Dataset Maestro
- **Distribución de Clases (n=11,606)**:
    - **Sanos (CN)**: 3,922 muestras (33.8%).
    - **Deterioro Cognitivo Leve (MCI)**: 3,761 muestras (32.4%).
    - **Alzheimer (AD)**: 3,923 muestras (33.8%).
- **Equilibrio Estadístico**: El dataset presenta un **balance excepcional**, permitiendo el entrenamiento de redes neuronales sin sesgos por clase.
- **Correlación de Imagen**: Las secuencias identificadas (MPRAGE y Accelerated Sagittal IR-FSPGR) garantizan la visualización estándar del hipocampo, fundamental para la arquitectura NeuroNet-Fusion.

### Análisis Volumétrico (3D)
- **Extracción Masiva**: Implementación de un pipeline de extracción automatizada desde 10 archivos ZIP (~150GB), recuperando volúmenes completos (series DICOM).
- **Integridad de Datos**: Verificación de la presencia de 160-200 cortes por serie para reconstrucción 3D perfecta.
- **Hardware Utilizado**: Procesamiento acelerado mediante **NVIDIA RTX 4070**, permitiendo el escaneo de miles de archivos en tiempos reducidos.
