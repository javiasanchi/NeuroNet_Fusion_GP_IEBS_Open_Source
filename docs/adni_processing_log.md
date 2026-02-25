# Fase de Procesamiento de Datos ADNI - NeuroNet-Fusion

## 1. Identificación y Contrastación de Sujetos
- **Sujetos en Imágenes (ZIP):** 2,767 sujetos únicos extraídos de 10 paquetes ZIP.
- **Sujetos en Metadatos (ADNIMERGE):** 4,946 sujetos registrados, pero solo 2,049 coincidían con el lote de imágenes inicial.
- **Ampliación con DXSUM:** Se incorporó la tabla primaria `DXSUM_16Feb2026.csv`, logrando identificar el diagnóstico para **2,746 de los 2,767 sujetos** (99% de cobertura).

## 2. Refinamiento del Dataset
Para el entrenamiento del modelo, se aplicaron los siguientes criterios de selección:
- **Modalidad:** MRI (Resonancia Magnética).
- **Contraste:** T1-weighted (preferencia por MPRAGE o Accelerated Sagittal IR-FSPGR).
- **Tipo de Imagen:** Original (sin pre-procesamiento previo externo para mantener homogeneidad en nuestro pipeline).
- **Limpieza:** Eliminación de "Localizers" y capturas de baja resolución.
- **Redundancia:** Selección de una única imagen representativa por visita de cada sujeto.

**Resultado Final del Refinamiento:**
- **Total de imágenes seleccionadas:** 11,606.
- **Distribución por Clase:**
  - Alzheimer (AD): 3,923 imágenes.
  - Sano (CN): 3,922 imágenes.
  - Deterioro Cognitivo Leve (MCI): 3,761 imágenes.

## 3. Estrategia de Extracción y Procesamiento Masivo
- **Extracción Volumétrica (3D):** Se ha pasado de una extracción 2D a una 3D, recopilando todas las rebanadas DICOM de cada serie seleccionada (aprox. 160-200 cortes por paciente).
- **Conversión a NIfTI:** Uso de la librería `dicom2nifti` para consolidar los cortes en volúmenes 3D comprimidos (.nii.gz), facilitando el manejo por redes neuronales convolucionales 3D.
- **Normalización de Intensidad:** Aplicación de normalización Z-score (centrada en media 0 y desviación estándar 1, excluyendo el fondo) para asegurar que el modelo sea robusto ante variaciones en la intensidad de los diferentes escáneres.
- **Reorientación:** Aseguramiento de la orientación estándar (RAS) para todos los volúmenes.

---
*Documento actualizado al 16 de febrero de 2026 por el asistente Antigravity.*
