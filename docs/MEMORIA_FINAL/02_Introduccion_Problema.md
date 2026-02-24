# FASE 2 — INTRODUCCIÓN: EL PROBLEMA Y LA MOTIVACIÓN

---

## 2.1 Contexto Clínico y Epidemiológico

La enfermedad de Alzheimer (EA) es un trastorno neurodegenerativo crónico de etiología multifactorial que destruye progresiva e irreversiblemente las neuronas del sistema nervioso central. Representa entre el **60% y el 80%** de todos los casos de demencia a nivel mundial (Alzheimer's Association, 2024) y es la **séptima causa de muerte** en los países desarrollados.

Los datos epidemiológicos globales son alarmantes:

| Indicador | Cifra Global (2024) |
|:---|:---|
| Personas con demencia en el mundo | 55 millones |
| Nuevos casos anuales | 10 millones |
| Coste económico global | 1,3 billones de USD |
| Pacientes sin diagnóstico formal | ~75% en países de renta media-baja |
| Proyección de prevalencia (2050) | 139 millones |

*Fuente: World Alzheimer Report 2024, Alzheimer's Disease International.*

---

## 2.2 El Desafío del Diagnóstico Precoz

La fisiopatología de la EA sigue una **cascada temporal** bien definida que precede en **15-20 años** a los síntomas clínicos visibles:

```
[Fase Preclínica]          [Fase Prodrómica]       [Fase Demencia]
  ↓ Aβ42 en LCR           MCI (Deterioro          Alzheimer
  ↑ pTau en LCR           Cognitivo Leve)         Establecido
  Atrofia hipocampal
  
< -15 años               -5 a -2 años           Diagnóstico actual
```

El problema central es que el **diagnóstico convencional** ocurre en la fase de demencia establecida, cuando la pérdida neuronal es irreversible. Los métodos tradicionales dependen de:

1. **Evaluación neuropsicológica** (MMSE, CDR): subjetiva y operador-dependiente.
2. **Interpretación visual de MRI**: variabilidad interradiológica del 15-30%.
3. **PET amiloide**: gold standard biológico, pero coste de ~5.000€ por exploración y no disponible en la mayoría de hospitales.

---

## 2.3 La Brecha Tecnológica

Los modelos de IA previos para el diagnóstico de Alzheimer presentan **tres limitaciones críticas**:

| Limitación | Descripción |
|:---|:---|
| **Unimodalidad** | Solo analizan imagen MRI o solo datos clínicos, perdiendo la correlación entre ambos. |
| **2D vs. 3D** | Los modelos 2D analizan cortes aislados, perdiendo la continuidad espacial del hipocampo. |
| **Caja Negra** | Sin explicabilidad, los médicos no pueden validar ni confiar en la predicción. |

---

## 2.4 La Propuesta NeuroNet-Fusion

Frente a estas limitaciones, **NeuroNet-Fusion** se diseña con tres innovaciones centrales:

1. **Fusión Multimodal Profunda:** La red integra la información de imagen (ResNet50/3D-ResNet) con 14 biomarcadores clínicos (rama MLP), generando un diagnóstico contextualmente enriquecido que ninguna fuente podría producir de forma aislada.

2. **Procesamiento Volumétrico 3D:** Las MRI se procesan como volúmenes completos (128×128×128 voxels), preservando la continuidad anatómica de estructuras críticas como el hipocampo, la corteza entorrinal y el lóbulo temporal medial.

3. **Explicabilidad Integrada:** El sistema genera mapas de calor Grad-CAM que muestran *dónde* mira el modelo y análisis SHAP que revelan *qué biomarcadores* determinan el diagnóstico, convirtiendo la IA en una "segunda opinión razonada" para el neurólogo.

---

## 2.5 Estructura de la Memoria

Esta memoria documenta el proyecto en **6 bloques temáticos** que cubren el ciclo de vida completo del sistema de inteligencia artificial:

- **Bloque I (Fases 1-3):** Cimentación — problema, objetivos y marco conceptual.
- **Bloque II (Fases 4-5):** Contexto científico — estado del arte y teoría.
- **Bloque III (Fases 6-8):** Ingeniería de datos — adquisición, preprocesado y características.
- **Bloque IV (Fases 9-11):** Desarrollo del modelo — benchmarking, arquitectura y entrenamiento.
- **Bloque V (Fases 12-13):** Validación — resultados y explicabilidad clínica.
- **Bloque VI (Fases 14-16):** Despliegue, conclusiones y bibliografía.
