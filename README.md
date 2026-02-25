<div align="center">

# 🧠 NeuroNet-Fusion: Precision Multimodal Diagnostic System for Alzheimer's Disease

### Artificial Intelligence Applied to Clinical decision Support Systems (CDSS)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Streamlit](https://badges.aleen42.com/src/streamlit.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**IEBS Digital School — Postgrado en Inteligencia Artificial Aplicada 2026**  
*Autor: Javier Sanchidrián Sanchidrián | Directora: Zaira Vicente Adame*

---

### [🚀 ACCEDER A LA APLICACIÓN EN VIVO](https://neuronet.iawordpress.com)
*Despliegue estable con SSL, Nginx y Docker*

---

> *"El Alzheimer nos roba la memoria. La Inteligencia Artificial nos devuelve el tiempo de reacción clínica."*

</div>

---

## 📝 Abstract (Resumen Académico)

**NeuroNet-Fusion** es un sistema avanzado de soporte a la decisión clínica (CDSS) diseñado para la detección precoz y estadificación biológica de la Enfermedad de Alzheimer. A diferencia de los enfoques tradicionales basados únicamente en imagen, este sistema implementa un modelo de **fusión multimodal tabular** que integra 14 biomarcadores clave: cognitiva (MMSE, CDR, FAQ), volumetría estructural refinada (MRI) y marcadores moleculares de líquido cefalorraquídeo (LCR).

Basado en el marco de investigación **NIA-AA 2018 (Sistema ATN)**, el sistema alcanza una **sensibilidad del 100%** en la detección de estadios moderados (AD), minimizando el riesgo de falsos negativos críticos. La implementación incluye un **Agente de IA Generativa** para la redacción de informes neurológicos estructurados y un escáner NLP para la ingesta automática de datos clínicos.

---

## 🔥 Características Destacadas (v2.0)

- **🏆 Motor de Inferencia:** Algoritmo XGBoost ultra-optimizado con una precisión global del **86.5%**.
- **🤖 Agente IA Agéntico:** Integración con GPT-4o y Gemini 2.0 Flash para interpretación clínica profunda.
- **🔍 Escáner NLP:** Extractor por procesamiento de lenguaje natural que convierte informes médicos de texto libre en biomarcadores estructurados.
- **🛡️ Marco ATN:** Clasificación automática A (Amiloide), T (Tau) y N (Neurodegeneración).
- **📦 Infraestructura de Grado Médico:** Contenerización con Docker y proxy inverso Nginx para alta disponibilidad y seguridad.

---

## 🏗️ Arquitectura Técnica

El sistema ha sido diseñado bajo una arquitectura de microservicios ligera que permite su integración en entornos hospitalarios (EHR) sin necesidad de infraestructura de computación GPU pesada:

![Diagrama de Arquitectura](reports/figures/grafico_10_4_arquitectura_final.png)

### Componentes Core:
1.  **Frontend:** Streamlit con diseño ultra-compacto "Single Screen" para eficiencia clínica.
2.  **Model Layer:** XGBoost Binary & Multiclass models (Joblib).
3.  **Knowledge Layer:** NIA-AA 2018 Evidence-based rules.
4.  **AI Layer:** Orchestration de LLMs (OpenAI/Google) para síntesis de informes.

---

## 📊 Resultados de la Investigación

Se realizó un benchmarking exhaustivo sobre **11,606 pacientes** de las cohortes ADNI y OASIS-3:

| Algoritmo | Accuracy | F1-Score | AUC-ROC |
|:---|:---:|:---:|:---:|
| **🏆 XGBoost (Producción)** | **86.5%** | **0.864** | **0.898** |
| LightGBM | 85.1% | 0.849 | 0.891 |
| CatBoost | 84.3% | 0.841 | 0.885 |
| Random Forest | 82.7% | 0.824 | 0.876 |

### Explicabilidad Clínica (SHAP)
El modelo no es una "caja negra". Mediante valores SHAP, se valida que los factores de mayor peso (MMSE, CDR, Hipocampo) coinciden con los protocolos clínicos internacionales más rigurosos.

---

## 🚀 Despliegue y Reproducibilidad

### Mediante Docker (Recomendado)
```bash
# Versión de portafolio para evaluadores
git clone https://github.com/javiasanc/NeuroNet_Fusion_GP_IEBS.git
cd NeuroNet_Fusion_GP_IEBS
docker-compose up -d --build
```

### Mediante Python Local
```bash
pip install -r requirements.txt
streamlit run src/app_diagnostics.py
```

---

## 📂 Estructura del Repositorio para Tutores

- `/src`: Código fuente de la estación de trabajo de diagnóstico.
- `/notebooks`: Cuadernos Jupyter con el proceso de EDA, Benchmarking y Optimización.
- `/docs`: Documentación técnica completa y Memoria del Proyecto.
- `/models`: El modelo final entrenado (`neuro_fusion_final_v1.joblib`).
- `/reports/figures`: Galerilla completa de métricas, SHAP y matrices de confusión.

---

## 📚 Bibliografía y Referencias
*   *Jack, C.R. et al. (2018). NIA-AA research framework: Toward a biological definition of Alzheimer's disease.*
*   *Alzheimer's Disease Neuroimaging Initiative (ADNI). adni.loni.usc.edu*
*   *Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.*

---

<div align="center">

**Proyecto Final de Postgrado — 2026**  
*Impulsando el diagnóstico de precisión mediante Inteligencia Artificial.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tu-perfil-si-quieres)

</div>
