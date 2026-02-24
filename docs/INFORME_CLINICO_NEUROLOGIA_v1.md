# INFORME TÉCNICO-CLÍNICO DEL PROYECTO NeuroNet-Fusion
## Para Revisión por el Equipo de Neurología

**Institución:** IEBS Business School — Postgrado en IA y Deep Learning  
**Versión del documento:** 1.0  
**Fecha:** 19 de Febrero de 2026  
**Responsable técnico:** Proyecto Global NeuroNet-Fusion  
**Destinatarios:** Equipo de Neurología — Consulta de revisión clínica

---

## RESUMEN EJECUTIVO

Este documento describe de forma integral el proyecto de investigación aplicada **NeuroNet-Fusion**, un sistema de inteligencia artificial diseñado para el apoyo al diagnóstico precoz de la Enfermedad de Alzheimer (EA) y el Deterioro Cognitivo Leve (MCI). Se solicita a los especialistas en neurología una revisión crítica de:

1. La **adecuación clínica** de las fuentes de datos utilizadas (ADNI, OASIS-3).
2. La **jerarquía de importancia de los datos**: Especial atención a los pesos relativos (Genética: Muy Alto en asintomáticos; Cognición: Muy Alto en MCI/AD; MRI: Alto en atrofia iniciada).
3. La **pertinencia de los biomarcadores** seleccionados (APOE, PRS, Herencia, LCR).
4. El **peso relativo de la imagen vs. biomarcadores**: Validación de que en asintomáticos la genética aporta ~85% del valor predictivo vs. ~14% de la MRI.
5. La **validez del etiquetado diagnóstico** de las clases (CN, MCI, AD).
6. Las **limitaciones clínicas identificadas** por el equipo técnico.

---

## SECCIÓN 1: CONTEXTO Y MOTIVACIÓN CLÍNICA

### 1.1 Problema Médico Objetivo

La Enfermedad de Alzheimer afecta aproximadamente al 5-8% de la población mayor de 65 años a nivel global, con proyecciones de triplicar los casos para 2050 (OMS, 2023). El desafío clínico principal radica en que el diagnóstico definitivo actual ocurre cuando la neurodegeneración ya es extensa, limitando severamente la eficacia de intervenciones preventivas.

El proyecto busca detectar patrones de atrofia cerebral **antes** de que la sintomatología cognitiva sea clínicamente evidente, actuando en la ventana terapéutica más favorable.

### 1.2 Hipótesis de Trabajo

La atrofia volumétrica del **hipocampo**, la **corteza entorrinal** y la **amígdala**, cuantificable mediante RM estructural T1, presenta patrones estadísticamente diferenciables entre los grupos:
- **CN** (Cognitivamente Normal)
- **MCI** (Deterioro Cognitivo Leve)
- **AD** (Enfermedad de Alzheimer establecida)

*⚠️ PREGUNTA AL EQUIPO DE NEUROLOGÍA (1): ¿Consideráis que la diferenciación MCI→AD es clínicamente estable mediante RM-T1 sin biomarcadores de LCR, o la superposición anatómica es demasiado elevada para un modelo de clasificación binaria?*

---

## SECCIÓN 2: FUENTES DE DATOS — DESCRIPCIÓN Y CRÍTICA

### 2.1 ADNI (Alzheimer's Disease Neuroimaging Initiative)

**Descripción:**
- Iniciativa multicéntrica pública financiada por el National Institutes of Health (NIH, EE.UU.)
- Inicio: 2003. Fases: ADNI-1, ADNI-GO, ADNI-2, ADNI-3 (activa).
- **Acceso:** Restringido, acceso aprobado mediante solicitud institucional.
- **Portal:** ida.loni.usc.edu

**Inventario actual del proyecto:**

| Clase | Volúmenes NIfTI ensamblados | Porcentaje |
|:---|:---:|:---:|
| AD (Alzheimer confirmado) | 251 | 30.4% |
| CN (Control Sano) | 294 | 35.6% |
| MCI (Deterioro Cognitivo Leve) | 280 | 34.0% |
| **TOTAL** | **825** | **100%** |

**Distribución del dataset maestro completo (antes del subconjunto ensamblado):**

El metadato del ADNI que el proyecto posee identifica **11.606 imágenes representativas** (una por visita para evitar redundancia temporal), con la siguiente distribución:
- CN: 3.922 (33.8%)
- MCI: 3.761 (32.4%)
- AD: 3.923 (33.8%)

> **Cobertura diagnóstica**: 99.2% de los sujetos identificados tienen diagnóstico vinculado, validado desde la tabla primaria `DXSUM_ADNIALL.csv`.

**Características técnicas de la imagen ADNI:**
- Tipo de secuencia: **T1-weighted MPRAGE** (Magnetization Prepared Rapid Gradient Echo) y **Accelerated Sagittal IR-FSPGR**
- Intensidad magnética: 1.5T y 3.0 Tesla (mix de equipos)
- Formato original: **DICOM** → convertido a **NIfTI** (.nii.gz)
- Resolución típica de volumen: 176×240×256 vóxeles
- Resolución de trabajo estandarizada: **96×96×96 vóxeles** (reescalado isótropo)

*⚠️ PREGUNTA AL EQUIPO (2): El dataset mezcla imágenes de campos magnéticos de 1.5T y 3.0T. ¿Consideráis que esta heterogeneidad es clínicamente relevante para la consistencia del análisis volumétrico del hipocampo? ¿Debería estratificarse el análisis por intensidad de campo?*

### 2.2 OASIS-3 (Open Access Series of Imaging Studies)

**Descripción:**
- Proyecto de la Universidad de Washington (WU), acceso completamente abierto.
- Contiene datos longitudinales de sujetos mayores con y sin demencia.
- Datos disponibles: RM cerebral, PET amiloide, LCR, pruebas cognitivas.
- **Estado en el proyecto:** En proceso de integración. Se ha descargado el índice de sujetos (`OASIS3_MR_json.csv`) pero los volúmenes DICOM aún no han sido procesados.

*⚠️ PREGUNTA AL EQUIPO (3): ¿El protocolo de adquisición de OASIS-3 es lo suficientemente compatible con ADNI para combinar ambos datasets sin sesgos de adquisición (batch effect)? ¿Recomendáis armonización ComBat o similar?*

---

## SECCIÓN 3: BIOMARCADORES UTILIZADOS — DESCRIPCIÓN Y PERTINENCIA

### 3.1 Biomarcadores Neuropsicológicos (Tests Cognitivos)

| Biomarcador | Nombre completo | Escala | Uso en el proyecto |
|:---|:---|:---:|:---|
| **MMSE** | Mini-Mental State Examination | 0–30 (Mayor = mejor) | Variable de entrada en rama clínica del modelo |
| **ADAS-13** | Alzheimer's Disease Assessment Scale (13 ítems) | 0–85 (Mayor = peor) | Variable de entrada en rama clínica |
| **CDRSB** | Clinical Dementia Rating Sum of Boxes | 0–18 (Mayor = peor) | Variable auxiliar de validación de etiqueta |

*⚠️ PREGUNTA AL EQUIPO (4): ¿El MMSE es suficientemente sensible para el estadio MCI, o recomendáis incluir el MoCA (Montreal Cognitive Assessment) que tiene mayor sensibilidad en deterioro leve? El ADNI incluye datos de MoCA en fases avanzadas.*

### 3.2 Biomarcadores de LCR (Líquido Cefalorraquídeo)

| Biomarcador | Descripción | Valores de referencia orientativos | Disponibilidad en dataset |
|:---|:---|:---:|:---:|
| **Aβ42 (ABETA)** | Amiloide-beta 1-42. Reducido en EA por depósito en placas. | >800 pg/mL = normal | Disponible en subconjunto ADNI |
| **Tau total** | Proteína Tau total. Aumentada por neuroinflamación. | <300 pg/mL = normal | Disponible en subconjunto ADNI |
| **p-Tau181 (PTAU)** | Tau fosforilada. Marcador de degeneración neurofibrilar. | <60 pg/mL = normal | Disponible en subconjunto ADNI |
| **Ratio Aβ42/Tau** | Índice compuesto. Ratio < 1.0 indica patología amiloide. | >1.0 = normal | Calculado en preprocesamiento |

**Estado actual:** Los biomarcadores de LCR están disponibles en el metadato ADNI pero **no han sido integrados en el modelo actual**. El modelo en entrenamiento solo usa imágenes MRI. La fusión multimodal (imagen + LCR + tests cognitivos) está planificada para la fase siguiente.

*⚠️ PREGUNTA AL EQUIPO (5): ¿Cuál es vuestra opinión sobre el valor añadido real del ratio Aβ42/Tau vs. Tau total para diferenciar MCI-amnésico de MCI-no amnésico? ¿El modelo debería priorizar p-Tau181 sobre Tau total como indicador independiente?*

### 3.3 Biomarcadores Genéticos

| Biomarcador | Descripción | Estado en el proyecto |
|:---|:---|:---:|
| **APOE ε4** | Alelo de mayor riesgo genético para EA esporádica | Disponible en metadata ADNI, no integrado aún |
| **APOE ε3/ε2** | Alelos protectores | Disponible en metadata ADNI |

*⚠️ PREGUNTA AL EQUIPO (6): El genotipo APOE ε4 tiene alta sensibilidad para EA tardía. ¿Debería incluirse como variable de entrada obligatoria en el modelo o tratarse como variable de estratificación para análisis de subgrupos?*

---

## SECCIÓN 4: TIPO DE RESONANCIA MAGNÉTICA — DESCRIPCIÓN TÉCNICA

### 4.1 Secuencias MRI Utilizadas en ADNI

Las secuencias disponibles y efectivamente utilizadas en el proyecto son:

| Secuencia | Nombre técnico | Campo | Uso principal |
|:---|:---|:---:|:---|
| **MPRAGE** | Magnetization Prepared Rapid Gradient Echo | 3.0T | Volumetría cerebral estructural. Estándar de facto para demencia |
| **IR-FSPGR** | Inversion Recovery Fast Spoiled Gradient Echo | 1.5T | Alternativa de alta resolución T1 en equipos 1.5T |

**Parámetros típicos (MPRAGE 3T ADNI-3):**
- TR: 2300 ms
- TE: 2.95 ms
- TI: 900 ms
- Ángulo de giro: 9°
- Resolución: 1×1×1 mm (isótropo)
- FOV: 256×240×176 mm

*⚠️ PREGUNTA AL EQUIPO (7): ¿La RM T1 estructural (MPRAGE) es suficiente como modalidad única para el objetivo del proyecto, o consideráis que la RM de difusión (DTI) o el PET amiloide son imprescindibles para diferenciar MCI productivo de MCI no-productivo?*

### 4.2 Lo que NO se utiliza (y debería considerarse)

| Modalidad | Descripción | Por qué no se usa actualmente |
|:---|:---|:---|
| **RM funcional (fMRI)** | Actividad neural en reposo (resting-state) | Complejidad de preprocesamiento (FSL/SPM) fuera del alcance actual |
| **PET Amiloide** | Carga de placas amiloides (FDG-PET, Florbetapir) | Dataset disponible en ADNI pero no descargado |
| **PET Tau** | Distribución de ovillos neurofibrilares | Idem |
| **DTI** | Integridad de tractos de sustancia blanca | No disponible en el subconjunto descargado |

*⚠️ PREGUNTA AL EQUIPO (8): ¿Qué nivel de certeza diagnóstica esperáis de un modelo que solo usa RM-T1 sin PET amiloide? ¿Es clínicamente aceptable como herramienta de cribado o solo como apoyo diagnóstico?*

---

## SECCIÓN 5: PIPELINE DE PREPROCESAMIENTO DE IMAGEN

### 5.1 Flujo de procesamiento actual

```
DICOM (series 160-200 frames)
        ↓
    [dicom2nifti]   → Conversión a NIfTI (.nii.gz)
        ↓
    [Reorientación RAS]  → Alineación anatómica estándar Right-Anterior-Superior
        ↓
    [Normalización de intensidad]
        Percentil 1-99 en tejido (vóxeles > 0)
        Min-max → rango [0, 1]
        ↓
    [Reescalado isótropo]  → 96×96×96 vóxeles (modo 3D)
                           → 224×224 px por slice (modo 2.5D)
        ↓
    [Augmentación en entrenamiento]
        - Flip sagital aleatorio
        - Rotación ±10° 
        - Ruido gaussiano (σ=0.01)
        - Variación de intensidad ±10%
```

### 5.2 Lo que NO se aplica (y que puede ser relevante clínicamente)

| Paso estándar en neuroimagen | Estado | Impacto estimado |
|:---|:---:|:---|
| **Skull stripping** (eliminación del cráneo) | ❌ No aplicado | El cráneo introduce ruido para el modelo. FreeSurfer o BET (FSL) son el estándar |
| **Registro a espacio MNI** (normalización espacial) | ❌ No aplicado | Sin esto, las coordenadas anatómicas no son comparables entre sujetos |
| **Corrección de campo de sesgo N4** | ❌ No aplicado | Las inhomogeneidades del campo magnético introducen gradientes de intensidad artificiales |
| **Segmentación de hipocampo** | ❌ No aplicado | FreeSurfer/FastSurfer podría extraer métricas volumétricas directas |

*⚠️ PREGUNTA AL EQUIPO (9): Dados los recursos disponibles, ¿cuál de estos pasos de preprocesamiento consideráis IMPRESCINDIBLE para la validez clínica del modelo? ¿El skull stripping y la corrección N4 son suficientes para el cribado básico?*

---

## SECCIÓN 6: ARQUITECTURA DEL MODELO

### 6.1 Modelos evaluados en el benchmark

| Modelo | Tipo | Val. Accuracy (200 samples) | Observaciones |
|:---|:---:|:---:|:---|
| TriPlanar Fusion | Transfer Learning 2D | **57.3%** | Mejor resultado. ResNet50+DenseNet121 preentrenados |
| ResNet3D (16 filtros) | CNN 3D | 53.3% | Lento. Requiere >1000 muestras |
| ResNet3D-Deep (32f) | CNN 3D | 53.3% | Idem |
| DenseNet3D | CNN 3D | 53.3% | Sin ventaja over ResNet3D |
| Attention3D | CNN 3D | 48.0% | Alta varianza, inestable con pocas muestras |
| SVM (baseline) | Clásico | 52.0% | Referencia. Competitivo con redes 3D pequeñas |

**Mejor resultado histórico registrado:** 86.5% accuracy (con modelo anterior en subconjunto de datos diferente — necesita revalidación).

**Estado actual del entrenamiento (19/02/2026):**
- Modelo en ejecución: **TriPlanar Fusion** (ResNet50 + DenseNet121, preentrenados ImageNet)
- Épocas completadas: ~20/100
- Mejor accuracy actual: **43.4%** (en proceso de convergencia)
- GPU: RTX 4070 Ti Super (17 GB VRAM)

### 6.2 Arquitectura TriPlanar Fusion (modelo actual)

```
Volumen MRI 3D (96×96×96)
        ↓
  Extracción de 3 vistas anatómicas:
  ┌─────────┬──────────┬───────────┐
  │ AXIAL   │ CORONAL  │ SAGITTAL  │
  │(superior│(anterior)│ (lateral) │
  │ ↓)      │   ↓)     │    ↓)     │
  └─────────┴──────────┴───────────┘
        ↓ (cada vista: 224×224 px, 3 canales RGB)
  ┌──────────────────────────────────┐
  │    ResNet50 (preentrenado)       │ → 2048 features por vista
  │    DenseNet121 (preentrenado)    │ → 1024 features por vista
  └──────────────────────────────────┘
        ↓ Concatenación de 3 vistas = (2048+1024) × 3 = 9216 features
  ┌──────────────────────────────────┐
  │       Clasificador               │
  │  9216 → 512 → 128 → 3 clases    │
  │  (LayerNorm + Dropout 0.5)       │
  └──────────────────────────────────┘
        ↓
  Diagnóstico: CN / MCI / AD
```

---

## SECCIÓN 7: LIMITACIONES IDENTIFICADAS POR EL EQUIPO TÉCNICO

Las siguientes limitaciones han sido identificadas internamente y se solicita valoración clínica de su relevancia:

### 7.1 Limitaciones de Datos

| Limitación | Descripción | Severidad técnica | Pregunta al equipo |
|:---|:---|:---:|:---|
| **Tamaño de muestra reducido** | 825 volúmenes ensamblados de ~11.600 disponibles en ADNI. La descarga completa requiere ~150GB adicionales. | 🔴 Alta | ¿Es 825 suficiente para un estudio piloto o el modelo necesita el dataset completo para ser clínicamente relevante? |
| **Sin skull stripping** | El cráneo y el cuello están presentes en algunas imágenes, añadiendo ruido irrelevante. | 🟡 Media | ¿Qué porcentaje de lecturas radiológicas quedarían invalidadas clínicamente por este motivo? |
| **Mix 1.5T / 3.0T** | Diferentes intensidades de campo → diferente SNR y contraste T1. | 🟡 Media | ¿Recomendáis análisis separados por intensidad de campo? |
| **Sin registro MNI** | Los volúmenes no están en el mismo espacio estándar. El hipocampo no está en la misma posición en todos los sujetos. | 🔴 Alta | ¿Es el registro espacial imprescindible para un modelo de clasificación? ¿O la CNN aprende a ser invariante? |
| **Datos de LCR no integrados** | Tau, p-Tau, Aβ42 disponibles pero no usados en el modelo actual. | 🟡 Media | ¿Cuánto mejoraría el poder diagnóstico integrar LCR? |

### 7.2 Limitaciones del Modelo

| Limitación | Descripción |
|:---|:---|
| **Clasificación estática (no longitudinal)** | El modelo clasifica cada visita de forma independiente, sin aprovechar la evolución temporal del paciente |
| **Sin cuantificación de incertidumbre** | No se calcula intervalo de confianza — el modelo no sabe cuándo "no sabe" |
| **No interpretable por región** | Actualmente sin Grad-CAM activado — el modelo no indica qué zona del cerebro influye en el diagnóstico |
| **No validado con datos externos** | Solo entrenado y evaluado con ADNI — sin test en datos de hospital real |

---

## SECCIÓN 8: COMPARATIVA CON ESTÁNDARES PUBLICADOS

### 8.1 Rendimiento de referencia en literatura (RM-T1, clasificación CN/MCI/AD)

| Referencia | Dataset | Metodología | Accuracy |
|:---|:---|:---|:---:|
| Hosseini et al. (2024) | ADNI | 3D-CNN + attention | 87.2% |
| Liu et al. (2023) | ADNI + AIBL | ResNet50 2.5D | 83.4% |
| Zhang et al. (2022) | ADNI | Multi-scale CNN | 81.7% |
| **NeuroNet-Fusion (objetivo)** | ADNI | TriPlanar ResNet50+DenseNet121 | **86.5% (meta)** |
| **NeuroNet-Fusion (actual)** | ADNI | TriPlanar (entrenamiento en curso) | **~43-57% (provisional)** |

*Nota: Los resultados actuales son preliminares. El modelo lleva ~20 de 100 épocas de entrenamiento.*

---

## SECCIÓN 9: PREGUNTAS CONSOLIDADAS PARA EL EQUIPO DE NEUROLOGÍA

A continuación se recopilan todas las preguntas de revisión clínica identificadas en este informe:

| N° | Pregunta |
|:---:|:---|
| 1 | ¿Es la RM-T1 sin LCR suficiente para diferenciar MCI de CN con relevancia clínica? |
| 2 | ¿El mix 1.5T/3.0T invalida la comparabilidad o es aceptable con normalización de intensidad? |
| 3 | ¿La combinación ADNI + OASIS-3 sin armonización ComBat introduce sesgos de adquisición relevantes? |
| 4 | ¿El MMSE es suficientemente sensible para MCI o se debería incluir MoCA? |
| 5 | ¿Debería priorizarse p-Tau181 sobre Tau total? ¿El ratio Aβ42/Tau es imprescindible? |
| 6 | ¿APOE ε4 debe ser variable obligatoria de entrada o solo de estratificación? |
| 7 | ¿RM-T1 sola es suficiente como modalidad única o se necesita PET amiloide para uso clínico real? |
| 8 | ¿Qué nivel de certeza clínica se puede atribuir a un modelo solo con RM-T1? |
| 9 | ¿Cuáles son los pasos de preprocesamiento imprescindibles: skull stripping, N4, registro MNI? |
| 10 | ¿La clasificación triclase (CN/MCI/AD) es clínicamente útil o debería ser un continuo de riesgo? |

---

## SECCIÓN 10: PRÓXIMOS PASOS TÉCNICOS PLANIFICADOS

| Fase | Acción | Estado |
|:---|:---|:---:|
| Entrenamiento TriPlanar GPU | 100 épocas con RTX 4070 Ti Super | 🔄 En curso |
| Integración de biomarcadores LCR | Añadir Tau/pTau/Abeta al modelo | ⏳ Planificado |
| Skull stripping automático | FSL BET o HD-BET | ⏳ Pendiente de decisión |
| Registro MNI | ANTs o FSL FLIRT | ⏳ Pendiente de decisión |
| Grad-CAM / interpretabilidad | Mapas de calor sobre anatomía | ⏳ Planificado (post-entrenamiento) |
| Validación externa | Hospital real o dataset independiente | ⏳ Largo plazo |

---

## SECCIÓN 11: JERARQUÍA DE IMPORTANCIA Y PESOS RELATIVOS (ESTRATEGIA ACTUALIZADA)

Tras una auditoría interna y revisión profunda de la capacidad predictiva de cada modalidad, el proyecto ha determinado que los pesos de entrenamiento no deben ser estáticos, sino adaptativos según el estadio clínico.

### 11.1 Jerarquía de Pesos por Modalidad
| Modalidad | Peso Relativo | Función Crítica en el Modelo |
| :--- | :--- | :--- |
| **Genética (APOE/PRS)** | **Muy Alto (Fase Preclínica)** | Predicción a largo plazo (hasta 8 años antes). |
| **Tests Cognitivos (ADAS/MMSE)**| **Muy Alto (Fase MCI/AD)** | Mejor predictor de la conversión a corto plazo (18 meses). |
| **Resonancia Magnética (MRI)** | **Alto** | Identificación de atrofia estructural ya iniciada (fenotipo). |
| **Analíticas LCR/Plasma** | **Alto** | Validación molecular de la patología amiloide y tau. |
| **Herencia Familiar** | **Medio/Contextual** | Estratificación de riesgo y marcador genético subrogado. |

### 11.2 Análisis Predictivo según Estadio
- **Pacientes Asintomáticos (CN)**: La genética domina. Estudios con ADNI demuestran una precisión de **0.857** basada en genética frente a **0.143** de la MRI en esta fase.
    - **APOE ε4**: Factor categórico. Homocigotos tienen riesgo x15.
    - **PRS (Puntuación de Riesgo Poligénico)**: Extiende la ventana de detección antes de cambios anatómicos visibles.
- **Conversión (MCI)**: El peso se desplaza a la cognición y analítica.
    - **ADAS-13/MMSE**: Elevan la precisión del modelo de 60.4% a 78.8% al combinarse con MRI.
    - **Ratio Aβ42/p-Tau181**: Gold Standard biológico para diferenciar sMCI de pMCI.
- **Atrofia Regional (MRI)**: La IA prioriza el **hipocampo**, el lóbulo temporal medio y la dilatación de los ventrículos laterales.

### 11.3 Configuración de Entrenamiento Recomendada
El modelo evolucionará hacia una **Fusión Multimodal con Pesos Adaptativos**:
1. **Entrada de Imagen**: 3D CNN (3D-ResNet) para capturar continuidad espacial de la atrofia.
2. **Canal de Meta-datos**: GNN (Graph Neural Networks) para modelar la herencia familiar como nodos con carga genética compartida.
3. **Mecanismo de Cross-Attention**: Los marcadores clínicos (ej. MMSE bajo) guiarán al modelo de imagen hacia la corteza entorrinal.
4. **Función de Pérdida**: Focal Loss o Joint Loss con Clustering para agrupar riesgos asintomáticos similares.

---

## SECCIÓN 12: PREGUNTAS CONSOLIDADAS (ACTUALIZADO)

| N° | Pregunta |
|:---:|:---|
| 11 | ¿Qué peso (porcentaje) le asignaríais en vuestra práctica clínica a la RM vs. Antecedentes Familiares en un paciente con MMSE > 26? |
| 12 | ¿Consideráis que un modelo que priorice la analítica de LCR y Genética sobre la imagen es más "leal" a la progresión biológica de la enfermedad? |
| 13 | ¿Es aceptable para el neurólogo que el modelo cambie sus "atenciones" (pesos) dinámicamente según la fase detectada del paciente? |
| 14 | ¿El uso de PRS (Riesgo Poligénico) se percibe como una herramienta útil en consulta o genera dilemas éticos por su carácter predictivo a largo plazo? |

---

## REFERENCIAS

1. Jack CR Jr, et al. "The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI methods." *J Magn Reson Imaging.* 2008.
2. LaMontagne PJ, et al. "OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset." *medRxiv.* 2019.
3. Hosseini MP, et al. "Multimodal deep learning for Alzheimer's disease dementia assessment." *Nat Commun.* 2023.
4. McKhann GM, et al. "The diagnosis of dementia due to Alzheimer's disease." *Alzheimers Dement.* 2011.
5. Jack CR, et al. "NIA-AA Research Framework: Toward a biological definition of Alzheimer's disease." *Alzheimers Dement.* 2018.

---

*Documento generado el 19/02/2026. Versión preliminar — sujeta a revisión por el equipo de neurología.*  
*Proyecto NeuroNet-Fusion | IEBS Business School | Postgrado en IA y Deep Learning*
