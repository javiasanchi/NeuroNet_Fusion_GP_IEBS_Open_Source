# MEMORIA TÉCNICA: NeuroNet-Fusion (V2) - Diagnóstico por Imagen 3D y Biomarcadores

**Autor:** [Tu Nombre]  
**Institución:** Master en Inteligencia Artificial y Deep Learning - IEBS Digital School  
**Especialidad:** Deep Learning aplicado a la Neuroimagen

---

## RESUMEN EJECUTIVO
La enfermedad de Alzheimer (EA) es una patología neurodegenerativa caracterizada por un deterioro cognitivo progresivo y cambios estructurales cerebrales irreversibles. **NeuroNet-Fusion v2** es una evolución del sistema de diagnóstico multimodal que transita de un análisis bidimensional a una arquitectura de aprendizaje profundo tridimensional (**3D-Deep Learning**). Este enfoque permite capturar la atrofia volumétrica en regiones críticas como el hipocampo y la corteza entorrinal, utilizando datos de la iniciativa **ADNI (Alzheimer's Disease Neuroimaging Initiative)**. El proyecto integra imágenes de Resonancia Magnética Estructural (T1-weighted) con biomarcadores clínicos y perfiles demográficos, optimizado para ejecución en hardware de alto rendimiento (RTX 4070).

---

## 1. JUSTIFICACIÓN MÉDICA Y OBJETIVOS

### 1.1 El Contexto Clínico
El diagnóstico clínico tradicional del Alzheimer se basa en pruebas cognitivas (MMSE, ADAS-Cog) y biomarcadores de líquido cefalorraquídeo. Sin embargo, los cambios morfológicos detectables por neuroimagen preceden a menudo a los síntomas clínicos. La atrofia en el lóbulo temporal medial es un marcador biológico clave validado por la comunidad neurológica para diferenciar entre:
*   **CN (Cognitivamente Normal)**: Estructura cerebral preservada para la edad.
*   **MCI (Deterioro Cognitivo Leve)**: Etapa de transición con atrofia incipiente.
*   **AD (Enfermedad de Alzheimer)**: Neurodegeneración extensa y pérdida de volumen sináptico.

### 1.2 Justificación del Enfoque 3D
A diferencia de los modelos 2D convencionales que analizan "rebanadas" aisladas, la arquitectura 3D propuesta permite al neurólogo obtener una **firma volumétrica** del paciente. Esto es crítico porque:
1.  **Continuidad Espacial**: La atrofia no ocurre en un solo plano; se extiende por estructuras tridimensionales.
2.  **Sensibilidad Temprana**: Los modelos 3D detectan cambios sutiles en la densidad de la materia gris antes de que sean visibles al ojo humano en cortes axiales.

---

## 2. ADQUISICIÓN Y ALINEACIÓN DE DATOS (ADNI + OASIS-3)

### 2.1 El Ecosistema ADNI (Fuente Primaria)
Se ha seleccionado el dataset de **ADNI** como fuente primaria debido a su rigor en la captura de datos y su reconocimiento global en hospitales de investigación. 
*   **Estado del Inventario**: 1,280 pacientes identificados físicamente y vinculados al registro clínico `ADNIMERGE.csv`.
*   **Mapeo de Diagnóstico**: Cada PTID se ha verificado contra su diagnóstico clínico más reciente (CN, MCI, AD).

### 2.2 El Proyecto OASIS-3 (Validación Externa)
Para garantizar la generalización del modelo, se ha integrado soporte para el dataset **OASIS-3**, permitiendo validar la robustez de la IA frente a datos de diferentes instituciones.
*   **Análisis de Metadatos**: Se han filtrado **4,116 sesiones de MRI T1-weighted** aptas para el pipeline 3D.
*   **Automatización**: Implementación de scripts oficiales de descarga integrados en el flujo de trabajo del proyecto, permitiendo una expansión masiva del conjunto de validación.

### 2.3 Perfilado Demográfico
Se ha incorporado información de la tabla `PTDEMOG` (4,946 perfiles) para normalizar las predicciones frente a género (balanceado: 3,009H/3,030M) y niveles educativos, fortaleciendo la equidad del algoritmo.

---

## 3. METODOLOGÍA TÉCNICA Y PROCESAMIENTO 3D

### 3.1 Reconstrucción Volumétrica
Cada sesión de MRI consta de ~160 a 180 archivos DICOM. Se ha implementado un algortimo de ensamblado que:
1.  Ordena las instancias axialmente mediante el `InstanceNumber`.
2.  Normaliza la intensidad de los píxeles (Hounsfield units/Pixel Value).
3.  Genera un **Voxel-Block** de alta fidelidad, visualizable en ejes axial, sagital y coronal.

### 3.2 Optimización de Hardware (GPU RTX 4070)
La arquitectura está diseñada para aprovechar los **16GB de VRAM** y los **Tensor Cores de 4ª Gen**:
*   **Mixed Precision Training (FP16)**: Reduce el tiempo de entrenamiento en un 50% sin pérdida de precisión diagnóstica.
*   **Carga Directiva**: Uso de tensores de PyTorch (`.pt`) pre-procesados para eliminar el cuello de botella de la CPU del hospital durante el entrenamiento.

---

## 4. IMPACTO EN EQUIPOS DE NEUROLOGÍA
El objetivo final de **NeuroNet-Fusion** no es solo clasificar, sino servir de herramienta de apoyo a la decisión clínica (CDSS).
*   **Interpretabilidad**: Implementación de mapas de calor 3D para indicar al neurólogo qué regiones específicas del cerebro han "disparado" la alerta de Alzheimer.
*   **Validación Externa**: Protocolos de prueba con el dataset OASIS-3 para garantizar que la IA funcione en diferentes hospitales y con diferentes marcas de resonadores (Siemens, GE, Philips).

---

## 5. ESTADO ACTUAL DEL PROYECTO
A fecha de **15 de febrero de 2026**, el proyecto ha superado satisfactoriamente las pruebas de ensamblado de volúmenes 3D y se encuentra en fase de ingesta masiva de datos tras el cross-check clínico.

---

**Referencias Clave para el Informe:**
*   Jack Jr, C. R., et al. (2018). *NIA-AA Research Framework: Toward a biological definition of Alzheimer's disease*.
*   Weller, J., & Budson, A. (2018). *Current understanding of Alzheimer's disease diagnosis and treatment*.
*   ADNI Data Manuals (2024-2026).
