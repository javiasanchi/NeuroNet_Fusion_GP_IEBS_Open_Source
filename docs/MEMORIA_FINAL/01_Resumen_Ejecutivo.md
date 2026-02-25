# FASE 1 — RESUMEN EJECUTIVO Y ABSTRACT

---

## Abstract (EN)

This project presents **NeuroNet-Fusion**, a multimodal clinical decision support system (CDSS) for the early detection of Alzheimer's Disease (AD). The system is built on 14 clinical, neuropsychological and molecular biomarkers derived from the ADNI and OASIS-3 cohorts — including cognitive scales (MMSE, CDR, FAQ), FreeSurfer-derived brain volumetrics normalized by intracranial volume (Hippocampus/ICV, Entorhinal/ICV), and CSF molecular markers (Aβ42, Tau, pTau-181) — and classifies patients into CN, MCI, or AD. A systematic benchmark of 12 algorithm families demonstrated that an optimized **XGBoost model** over tabular biomarkers achieves **86.5% accuracy** (F1=0.864, AUC-ROC=0.898), outperforming all CNN-based MRI approaches. The system is deployed as an interactive Streamlit web application generating structured neurological reports aligned with the ATN (Amyloid-Tau-Neurodegeneration) biomarker framework. Model transparency is provided via **SHAP (TreeSHAP)** feature importance analysis.

**Keywords:** Alzheimer's Disease, XGBoost, Clinical Biomarkers, ATN Framework, SHAP, Streamlit CDSS, ADNI, OASIS-3, Multimodal Benchmarking.

---

## Resumen (ES)

Este proyecto presenta **NeuroNet-Fusion**, un sistema de soporte a la decisión clínica (CDSS) multimodal para la detección temprana de la enfermedad de Alzheimer (EA). El sistema se basa en 14 biomarcadores clínicos, neuropsicológicos y moleculares derivados de las cohortes ADNI y OASIS-3 —incluyendo escalas cognitivas (MMSE, CDR, FAQ), volumetría cerebral derivada de FreeSurfer normalizada por el volumen intracraneal (Hipocampo/ICV, Entorrinal/ICV) y marcadores moleculares del LCR (Aβ42, Tau, pTau-181)— y clasifica a los pacientes en CN (Cognitivamente Normal), MCI (Deterioro Cognitivo Leve) o AD (Alzheimer). Un benchmarking sistemático de 12 familias de algoritmos demostró que un **modelo XGBoost** optimizado sobre biomarcadores tabulares alcanza una **precisión del 86.5%** (F1=0.864, AUC-ROC=0.898), superando a todos los enfoques de MRI basados en redes neuronales convolucionales (CNN). El sistema se despliega como una aplicación web interactiva en Streamlit que genera informes neurológicos estructurados alineados con el marco de biomarcadores ATN (Amiloide-Tau-Neurodegeneración). La transparencia del modelo se proporciona mediante el análisis de importancia de características **SHAP (TreeSHAP)**.

**Palabras Clave:** Enfermedad de Alzheimer, XGBoost, Biomarcadores Clínicos, Marco ATN, SHAP, CDSS Streamlit, ADNI, OASIS-3, Benchmarking Multimodal.

---

## Resumen Ejecutivo (ES)

La enfermedad de Alzheimer (EA) representa la primera causa de demencia a nivel mundial, afectando a más de **55 millones de personas** (OMS, 2024). Su naturaleza neurodegenerativa progresiva e irreversible hace que el diagnóstico tardío —cuando la sintomatología es ya evidente— sea clínicamente inútil para la intervención preventiva. El **reto central** de la neurología moderna es desplazar el diagnóstico hacia la fase prodrómica, cuando el daño estructural aún es parcial y las intervenciones pueden ralentizar la progresión.

**NeuroNet-Fusion** aborda este desafío mediante la construcción de un pipeline de Inteligencia Artificial de extremo a extremo que integra:

1. **Biomarcadores Neuropsicológicos:** Escalas validadas internacionalmente (MMSE, CDR, FAQ, ADAS-11) que cuantifican el deterioro cognitivo en múltiples dominios.
2. **Biomarcadores Estructurales (volumetría normalizada):** Medidas derivadas de la segmentación FreeSurfer sobre MRI T1-weighted: Hipocampo/ICV, Corteza Entorrinal/ICV, Región Temporal Media/ICV y Ventrículos/ICV. Estas medidas son **variables numéricas** en el dataset tabular, no imágenes directas.
3. **Biomarcadores Moleculares del LCR:** β-Amiloide-42, Tau total y pTau-181 obtenidos por punción lumbar, alineados con el marco ATN.
4. **Perfil Genético:** Estado portador del alelo APOE ε4, principal factor de riesgo hereditario de la EA esporádica.

> **Decisión de diseño clave:** El benchmarking empírico demostró que los modelos CNN sobre imagen MRI bruta alcanzan solo un 60% de accuracy frente al 86.5% del enfoque tabular. El **modelo final de producción opera exclusivamente sobre los 14 biomarcadores tabulares** — no ingiere imágenes como entrada.

---

### Resultados Clave

| Componente | Resultado |
|:---|:---|
| **Accuracy global (Test Set)** | **86.5%** |
| **F1-Score ponderado** | **0.864** |
| **AUC-ROC multiclase** | **0.898** |
| **Kappa de Cohen** | **0.797** (acuerdo sustancial) |
| **Sensibilidad — AD Moderado** | **100%** (0 falsos negativos) |
| **Sensibilidad — Estadio MCI** | **82%** |
| **Algoritmo final** | XGBoost optimizado (Optuna, 100 trials) |
| **Variables de entrada** | 14 biomarcadores tabulares (sin imagen directa) |
| **Aplicación clínica** | Streamlit CDSS + dictamen ATN descargable |
| **Integración NLP** | Agente Clínico (GPT-4o-mini) integrado |

![[Tabla 0.1 — Resumen de Resultados Clave de NeuroNet-Fusion]](../../reports/figures/tabla_resumen_ejecutivo.jpg)

---

### Impacto del Proyecto

- **Para la medicina:** Un sistema de apoyo a la decisión clínica (CDSS) que genera dictámenes neurológicos estructurados alineados con el marco ATN-NIA-AA 2018.
- **Para la academia:** Demostración rigorosa de un pipeline multimodal completo (datos → preprocesado → benchmarking → modelo → despliegue → explicabilidad).
- **Para el paciente:** Diagnóstico más precoz en la fase prodrómica (MCI), posibilitando intervenciones farmacológicas y preventivas antes de la demencia establecida.

---

> *"El Alzheimer nos roba la memoria. La Inteligencia Artificial nos devuelve el tiempo."*
