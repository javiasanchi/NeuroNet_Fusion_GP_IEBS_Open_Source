# Documento de Requerimientos del Proyecto: NeuroNet-Fusion

## 1. Introducción
El objetivo de este proyecto es desarrollar un sistema automatizado para la detección temprana de la enfermedad de Alzheimer mediante el análisis multimodal de imágenes de resonancia magnética (MRI) y datos clínicos tabulares.

## 2. Alcance
El sistema debe ser capaz de procesar datos de múltiples fuentes (ADNI, OASIS, Kaggle), realizar una limpieza exhaustiva, entrenar modelos de aprendizaje automático (clásicos y profundos) y proporcionar una justificación visual y estadística de sus predicciones.

## 3. Requerimientos Funcionales (RF)

### RF01: Ingesta Multidatos
- El sistema debe soportar la carga de archivos CSV con variables clínicas y carpetas de imágenes en formato JPG/PNG/NIfTI.
- Debe unificar pacientes de distintas fuentes bajo un identificador único (ID/RID).

### RF02: Preprocesamiento de Imágenes
- Normalización de intensidades (Z-score o ecualización de histograma).
- Redimensionamiento uniforme (224x224 para 2D).
- Data augmentation para modelos de Deep Learning (rotación, zoom, flip).

### RF03: Extracción de Características
- Cálculo de texturas (GLCM: contraste, homogeneidad, etc.).
- Descriptores locales (LBP).
- Obtención de embeddings profundos mediante redes preentrenadas (EfficientNet/ResNet).

### RF04: Entrenamiento de Modelos
- Soporte para modelos supervisados clásicos (SVM, RF, Gradient Boosting).
- Implementación de una red neuronal de fusión multimodal (CNN + MLP).
- Optimización de hiperparámetros mediante validación cruzada.

### RF05: Interpretabilidad y Explicabilidad
- Generación de mapas de saliencia (Grad-CAM) para localizar patologías en MRI.
- Análisis de importancia de variables clínicas mediante SHAP.

## 4. Requerimientos No Funcionales (RNF)

### RNF01: Reproducibilidad
- Todo el entorno debe estar definido en un archivo `requirements.txt`.
- Los scripts deben ser ejecutables de forma secuencial con resultados consistentes.

### RNF02: Portabilidad
- El sistema debe funcionar en sistemas Windows y Linux con soporte para CUDA.

### RNF03: Modularidad
- El código debe estar organizado en módulos independientes (data, features, models, reports).

### RNF04: Eficiencia
- El pipeline de datos debe ser eficiente en memoria (uso de DataLoaders y batches).
- Soporte para entrenamiento en precisión de 16-bits (Mixed Precision).
