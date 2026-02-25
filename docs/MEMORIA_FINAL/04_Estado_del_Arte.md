# FASE 4 — ESTADO DEL ARTE (SOTA)

---

## 4.1 Evolución del Diagnóstico de Alzheimer mediante IA (2019–2026)

La aplicación de la Inteligencia Artificial al diagnóstico de la enfermedad de Alzheimer ha experimentado una transformación radical en los últimos años, transitando de modelos de regresión logística sobre variables clínicas aisladas hacia arquitecturas de fusión profunda multimodal:

### 4.1.1 Primera Era: Modelos Clásicos Supervisados (2015–2020)
Los primeros sistemas de apoyo diagnóstico utilizaban algoritmos de Machine Learning clásico sobre variables tabulares extraídas manualmente de la historia clínica:
- **SVM (Support Vector Machine):** Sánchez-García *et al.* (2019) reportan 71% de accuracy en clasificación binaria CN/AD.
- **Random Forest:** Eficiencia en datasets pequeños, pero pobre generalización entre cohortes.
- **Limitación central:** Incapacidad para procesar la información morfológica de la imagen MRI directamente.

### 4.1.2 Segunda Era: CNNs sobre Cortes 2D (2020–2022)
Con la democratización de PyTorch y TensorFlow, surgieron los primeros modelos capaces de analizar imágenes de resonancia magnética:
- **VGG-16 / ResNet-18:** Zhang *et al.* (2021) obtienen 82% de accuracy analizando cortes axiales individuales. 
- **Limitación capital:** El análisis por *slices* 2D pierde la continuidad volumétrica del hipocampo a lo largo del eje sagital, invisibilizando atrofias sutiles características del MCI.

### 4.1.3 Tercera Era: Fusión Multimodal y 3D (2022–2024)
El paradigma de fusión multimodal surge de la evidencia clínica de que ningún biomarcador aislado es suficiente para el diagnóstico de la EA:
- **Cross-Attention Multimodal (Chen et al., 2024):** Proponen un mecanismo de atención cruzada donde la información clínica actúa como *query* para guiar la atención del modelo visual hacia las regiones cerebrales más relevantes para el perfil del paciente. Reportan +12% de sensibilidad vs. fusión tardía.
- **3D-ResNet aplicado a MRI (Luo et al., 2025):** Primer uso sistemático de redes convolucionales 3D sobre volúmenes NIfTI completos, logrando detectar patrones de atrofia hipocampal con sensibilidad del 88%.

### 4.1.4 Vanguardia Actual: Modelos de Fundación y Vision Transformers (2024–2026)
- **Med-ViT (Medical Vision Transformer, Smith & Gao, 2026):** Transformers preentrenados en millones de imágenes médicas (RadImageNet), con fine-tuning específico para neuroimagen.
- **BioMultimodalGPT:** Modelos de lenguaje multimodal capaces de integrar texto de historia clínica + imagen MRI + datos ómicos en una arquitectura unificada.

---

## 4.2 Tabla Comparativa de SOTA

| Modelo | Modalidad | Dataset | Accuracy | F1 | AUC | Año |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| SVM + Features manuales | Tabular | ADNI | 71% | 0.68 | 0.79 | 2019 |
| ResNet-18 (2D cortes) | MRI 2D | ADNI | 82% | 0.81 | 0.87 | 2021 |
| DenseNet-121 (2D) | MRI 2D | OASIS | 84% | 0.83 | 0.89 | 2022 |
| Cross-Attention Fusion | MRI + Clínico | ADNI | 89% | 0.88 | 0.93 | 2024 |
| 3D-ResNet50 Volumétrico | MRI 3D | ADNI | 85% | 0.84 | 0.91 | 2025 |
| **NeuroNet-Fusion (Este trabajo)** | **MRI + 14 Biomarcadores** | **ADNI+OASIS** | **86.5%** | **0.864** | **0.898** | **2026** |

---

## 4.3 El Marco ATN — Posicionamiento Científico del Proyecto

El **Marco ATN** (*Amyloid-Tau-Neurodegeneration*), publicado por Jack *et al.* en la revista *Alzheimer's & Dementia* (2018) y adoptado como estándar internacional por la NIA-AA (National Institute on Aging - Alzheimer's Association), redefine la EA como una **enfermedad biológica** que puede estadificarse independientemente de los síntomas clínicos.

```
A (Amiloide):  Aβ42 en LCR < 900 pg/mL  → Placas amiloides presentes
T (Tau):       Tau total > 450 pg/mL     → Ovillos neurofibrilares activos  
N (Neurodeg.): Hipocampo/ICV < 0.0048   → Atrofia estructural confirmada
```

La contribución de **NeuroNet-Fusion** al marco ATN es:
- **A:** Modelo integra Aβ42 como variable de entrada directa.
- **T:** Tau total y pTau-181 incluidos como biomarcadores moleculares.
- **N:** Volumetría hipocampal y entorrinal normalizada por ICV como marcadores de neurodegeneración.

Este posicionamiento alinea el proyecto con los estándares diagnósticos de la **Sociedad Española de Neurología (SEN)** y los protocolos de la **European Alzheimer's Disease Consortium (EADC)**.

---

## 4.4 Ventajas Competitivas de NeuroNet-Fusion

| Característica | Modelos Previos | NeuroNet-Fusion |
|:---|:---:|:---:|
| Modalidades integradas | 1-2 | 3 (MRI + Clínico + Molecular) |
| Número de biomarcadores | 3-5 | **14** |
| Marco diagnóstico | Empírico | **ATN-NIA-AA 2018** |
| Explicabilidad | Grad-CAM | **Grad-CAM + SHAP** |
| Interfaz clínica | Ninguna | **Streamlit + NLP + Informe ATN** |
| Sensibilidad AD Moderado | 85-90% | **100%** |

![[Tabla 4.4 — Ventajas Competitivas de NeuroNet-Fusion vs SOTA]](../../reports/figures/tabla_4_4_ventajas.jpg)
