# Fase 5: Preprocesamiento y Normalización (Pipeline 3D Profundo)

El pipeline de preprocesamiento se ha refinado para transformar datos crudos hospitalarios (DICOM) en volúmenes 3D optimizados para entrenamiento clínico.

## 5.1 Pipeline de Procesamiento Volumétrico
1.  **Conversión DICOM a NIfTI**: Uso de la librería `dicom2nifti` para consolidar series de cortes (160-200 frames) en volúmenes `nii.gz` unificados.
2.  **Alineación RAS (Right-Anterior-Superior)**: Reorientación automática de todos los volúmenes para asegurar la consistencia anatómica entre diferentes pacientes y centros.
3.  **Estado del Procesado (Hito Actual)**:
    - **Volúmenes Ensamblados**: 135 cerebros completos.
    - **Tasa de Éxito**: 100% de integridad en la conversión de la serie T1.
4.  **Normalización de Intensidad (Voxel-Wise)**:
    - **Z-Score Normalization**: Aplicación de la fórmula $z = \frac{x - \mu}{\sigma}$ sobre los voxels con señal (>0).
    - **Resultado**: Media 0 y Desviación Estándar 1, eliminando el sesgo producido por los diferentes voltajes de los imanes (1.5T y 3T).

## 5.2 Estrategia de Segmentación y Recorte
Para optimizar el entrenamiento en la **arquitectura NeuroNet-Fusion**:
- **Bounding Box Anatómico**: Recorte automático del espacio vacío (aire) alrededor del cráneo para centrar la red en el parénquima cerebral.
- **Redimensionamiento Uniforme**: Estandarización a **128x128x128 voxels**, permitiendo el uso de modelos de alta densidad (DenseNet3D) sin saturar los 16GB de VRAM de la GPU.

## 5.3 Normalización de Datos Tabulares
Estandarización de biomarcadores complementarios:
- **Scaling**: Uso de `StandardScaler` en variables críticas: edad, años de educación y puntuaciones MMSE/ADAS13.
- **Mapeo de Clases**: Codificación One-Hot para el diagnóstico, permitiendo al modelo de fusión (Cross-Attention) alinear correctamente la imagen con el riesgo clínico.

---
*Documentación técnica actualizada al nivel de investigación médica - Febrero 2026.*
