# FASE 13 — EXPLICABILIDAD E INTERPRETABILIDAD CLÍNICA

---

## 13.1 Filosofía: La "Caja de Cristal"

Un modelo de IA destinado al uso clínico no puede ser una **caja negra**. El médico necesita entender *por qué* el sistema emite un diagnóstico para poder validarlo, cuestionarlo y combinarlo con su criterio clínico.

NeuroNet-Fusion documenta **dos niveles de explicabilidad** correspondientes a las dos fases del proyecto:

| **Investigación** | Grad-CAM | CNN 2D (benchmarking) | Validación neuroanatómica del enfoque imagen |

![[Tabla 13.1 — Niveles de Explicabilidad e Interpretabilidad]](../../reports/figures/tabla_13_1_niveles.jpg)

> El modelo de producción (XGBoost tabular) **no usa Grad-CAM**. La sección 13.2 documenta Grad-CAM como resultado experimental sobre el modelo CNN que se evaluó en el benchmarking: su validación neuroanatómica refuerza la confianza en los biomarcadores volumétricos (Hipocampo/ICV, Entorrinal/ICV) que sí utiliza el modelo final como variables de entrada tabulares.

---

## 13.2 Grad-CAM: Validación Neuroanatómica del Modelo CNN (Benchmarking)

> **Contexto:** Grad-CAM se aplicó sobre el modelo **NeuroNetFusion CNN 2D** (ResNet50 + DenseNet121) que fue entrenado y evaluado durante la fase de benchmarking comparativo. Aunque este modelo no fue seleccionado para producción, sus mapas de saliencia confirman que la red aprendió a detectar las mismas estructuras que los neurorólogos identifican manualmente, fundamentando la validez clínica de los biomarcadores volumétricos que sí alimentan el modelo final.

![[Código 13.2.1 — Implementación de Grad-CAM en PyTorch (NeuroNetFusion)]](../../reports/figures/codigo_13_2_1_gradcam.jpg)

**Explicación técnica (13.2.1):**
El código implementa el algoritmo **Gradient-weighted Class Activation Mapping**, seleccionando la última capa convolucional de la ResNet50 como capa objetivo. Al calcular el gradiente de la puntuación de la clase predicha respecto a las activaciones de esta capa, el sistema identifica qué neuronas tienen mayor influencia positiva en la decisión, generando una máscara de importancia espacial que se proyecta sobre la MRI original.

![[Imagen 13.2.2 — Visualización Grad-CAM sobre MRI (CN vs MCI vs AD)]](../../reports/figures/gradcam_explainability.png)

**Explicación clínica (13.2.2):**
La visualización resultante permite una inspección cualitativa inmediata. Mientras que en sujetos sanos las activaciones son difusas, en pacientes con EA se observa una concentración de "calor" (puntos rojos) en el área del hipocampo y el sistema límbico, permitiendo al neurólogo verificar que el diagnóstico de la IA se basa en la atrofia neuroanatómica esperada y no en artefactos de la imagen.

### 13.2.2 Validación Neuroanatómica

Los mapas Grad-CAM fueron validados cualitativamente comparando las regiones de alta activación con los **atlas neuroanatómicos de referencia** (AAL3, Brodmann):

**Resultado 1 — Paciente CN (Sano):**
- Activación difusa, sin focalización patológica evidente.
- Alta varianza espacial → el modelo "no sabe qué buscar" (ausencia de atrofia).

**Resultado 2 — Paciente MCI (Deterioro Cognitivo Leve):**
- Activación moderada en región **CA1 del hipocampo** y **subículo**.
- Consistente con los primeros estadios de la escala de Braak (I-II).

**Resultado 3 — Paciente AD Moderado:**
- Activación intensa y bilateral en **hipocampo**, **corteza entorrinal** y **lóbulo temporal medio**.
- Patrón idéntico al reportado por el atlas de atrofia ADNI-MCI-AD (Hua *et al.*, 2023).

> *"El modelo aprende neuroanatomía sin que se la hayamos enseñado explícitamente. Esto valida la hipótesis de que los patrones de atrofia en la MRI son informativos para el diagnóstico."*

---

## 13.3 SHAP: Importancia de Biomarcadores

### 13.3.1 SHAP Beeswarm — Vista de Población

plt.savefig('reports/figures/shap_beeswarm_AD.png', dpi=150)
```

![[Código 13.3.1 — Generación de SHAP Beeswarm Plot para XGBoost]](../../reports/figures/codigo_13_3_1_shap_beeswarm.jpg)

**Explicación técnica (13.3.1):**
Se utiliza la implementación **TreeSHAP**, un algoritmo optimizado para modelos basados en árboles que permite calcular los **Valores SHAP (SHapley Additive exPlanations)** de forma exacta. El código genera una vista agregada (Beeswarm) donde cada punto representa a un paciente del conjunto de test. La posición en el eje X indica si el valor de un biomarcador aumentó o disminuyó la probabilidad de diagnóstico, mientras que el color representa el valor relativo del biomarcador (alto/bajo), permitiendo visualizar patrones globales de causalidad en toda la población estudiada.

![[Gráfico 13.3.1 — SHAP Beeswarm: Impacto de Biomarcadores en Diagnóstico AD]](../../reports/figures/shap_beeswarm_AD.png)

**Análisis de Hallazgos en Gráfico Beeswarm (Imagen 13.3.1):**

La visualización Beeswarm permite realizar una auditoría clínica de la lógica del modelo XGBoost. Cada punto representa a un paciente, situando su impacto en el eje horizontal (Valor SHAP):

*   **Variables Cognitivas (MMSE/CDR/FAQ):** Son los motores principales del diagnóstico. El **MMSE** muestra una correlación negativa perfecta: valores bajos (color azul) tienen SHAP positivo muy alto, indicando que el modelo "entiende" que un MMSE < 24 es el signo cardinal de Alzheimer.
*   **Biomarcadores Estructurales (Hipocampo/Entorrinal):** Los puntos azules (volúmenes normalizados bajos) se concentran en el lado derecho del gráfico. Esto confirma que la atrofia en estas regiones es identificada por la IA como una evidencia sólida de neurodegeneración.
*   **Biomarcadores Moleculares (LCR):** Se observa el fenómeno de "secuestro amiloide": valores bajos de **ABETA-42** (azul) aumentan el riesgo (SHAP positivo), mientras que para **TAU** y **pTAU**, son los valores elevados (rojo) los que impulsan la predicción de AD.
*   **Reserva Cognitiva y Genética:** Los años de educación desplazados a la izquierda (SHAP negativo) para valores altos (rojo) validan la teoría de la reserva cognitiva. El factor **APOE4** (alelo ε4) muestra un impacto positivo consistente, aunque de menor magnitud que las variables cognitivas directas.

**Interpretación del Beeswarm (clase AD):**

| Educación alta | ↓ P(AD) | Reserva cognitiva: años de educación actúan como factor protector |

![[Tabla 13.3.1 — Resumen de Impacto de Biomarcadores según SHAP]](../../reports/figures/tabla_13_3_1_shap.jpg)

### 13.3.2 SHAP Waterfall — Explicación Individual

    )
```

![[Código 13.3.2 — Generación de SHAP Waterfall Plot (Diagnóstico Individual)]](../../reports/figures/codigo_13_3_2_shap_patient.jpg)

**Análisis de Caso Individual:**
La visualización tipo **Waterfall** permite deconstruir la probabilidad final de un paciente. Partiendo del valor base (promedio de la población), cada biomarcador "empuja" la predicción hacia un diagnóstico u otro. En el ejemplo siguiente, el **MMSE** y la carga de **TAU** son los factores predominantes que confirman la transición a Alzheimer.

**Ejemplo: Paciente AD con MMSE=18, TAU=580, Hipocampo=0.0031**

![[Tabla 13.3.2 — Datos Clínicos de Paciente Ejemplo (Alzheimer)]](../../reports/figures/tabla_13_3_2_ejemplo_paciente.jpg)

```
SHAP Waterfall — Diagnóstico: Alzheimer Establecido (P=91.4%)

Base value (media población):            0.331
+ MMSE = 18          → +0.243  ████████████████████████
+ TAU = 580          → +0.147  ███████████████
+ Hipocampo = 0.0031 → +0.092  █████████
+ CDR = 2.0          → +0.068  ███████
+ ABETA = 680        → +0.041  ████
+ APOE4 = 1          → +0.028  ███
- EDUCYEARS = 16     → -0.031  ███ (factor protector)
────────────────────────────────────────────
Predicción final:                        0.919  ← P(AD) = 91.9%
```

---

## 13.4 Integración de Explicabilidad en la Aplicación

La aplicación Streamlit muestra automáticamente una interpretación narrative de los factores SHAP más importantes para cada predicción:

    narrative += get_clinical_explanation(feat, val)
```

![[Código 13.4 — Motor de Generación de Narrativa Clínica Streamlit]](../../reports/figures/codigo_13_4_narrativa.jpg)

**Explicación técnica (13.4):**
La implementación en la interfaz de usuario automatiza la traducción de los valores SHAP a lenguaje natural médico. El motor identifica los tres biomarcadores con mayor peso absoluto en la decisión actual (ya sea a favor o en contra del diagnóstico), determina su dirección (elevado/reducido) y consulta una base de conocimientos clínicos para generar una observación descriptiva. Este nivel de **interpretabilidad activa** permite que el informe final no solo entregue una probabilidad, sino también una justificación razonada de los hallazgos patológicos detectados.

---

## 13.5 Consistencia con Criterios Clínicos

La validación cruzada entre las predicciones del modelo y los criterios diagnósticos de la **Sociedad Española de Neurología (SEN)** confirma la alineación clínica:

| Educación = reserva cognitiva | Efecto negativo en P(AD) | ✅ 100% |

![[Tabla 13.5 — Validación de Concordancia con Criterios SEN]](../../reports/figures/tabla_13_5_sen.jpg)
