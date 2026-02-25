# Master Blueprint: Global Project IEBS - NeuroNet-Fusion 🧠

Este documento constituye la columna vertebral de la documentación del Proyecto Global de Fin de Posgrado. La estructura ha sido diseñada siguiendo los estándares de excelencia académica y profesional de **IEBS**, dividida en 5 bloques lógicos y 12 fases integrales.

---

## 🗂️ Estructura de Bloques y Fases

### BLOQUE 1: Definición y Análisis Conceptual
*   **[Fase 1: Definición del Problema](./Phase_01_Definicion_Problema.md)**: Identificación del reto médico y objetivos de la solución NeuroNet-Fusion.
*   **[Fase 2: Especificación de Requerimientos](./Phase_02_Requerimientos_Especificaciones.md)**: Requerimientos funcionales, no funcionales y pila tecnológica.
*   **[Fase 3: Estado del Arte y Propuesta](./Phase_03_Estado_del_Arte.md)**: Investigación SOTA (CNN, Cross-Attention) y justificación de la arquitectura.

### BLOQUE 2: Ingeniería de la Información (Data Engineering)
*   **[Fase 4: Adquisición y EDA](./Phase_04_Adquisicion_EDA.md)**: Origen de datos (ADNI, OASIS-3) y hallazgos exploratorios críticos.
*   **[Fase 5: Preprocesamiento y Normalización](./Phase_05_Preprocesamiento_Normalizacion.md)**: Pipelines de imagen (CLAHE) y estandarización tabular.
*   **[Fase 6: Ingeniería de Características](./Phase_06_Ingenieria_Caracteristicas.md)**: Extracción de Deep Embeddings y estrategia de fusión multimodal.

### BLOQUE 3: Desarrollo del Sistema de Inteligencia (Modeling)
*   **[Fase 7: Arquitectura del Modelo](./Phase_07_Arquitectura_Modelos.md)**: Diseño detallado de la red NeuroNet-Fusion (ResNet50 + MLP).
*   **[Fase 8: Entrenamiento y Optimización](./Phase_08_Entrenamiento_Optimizacion.md)**: Ciclo de entrenamiento, OneCycleLR y refinamiento de alta precisión.

### BLOQUE 4: Análisis de Resultados y Confianza Clínica
*   **[Fase 9: Análisis de Resultados](./Phase_09_Analisis_Resultados.md)**: Evaluación de métricas (Accuracy 86.5%, AUC 0.89) y matrices de confusión.
*   **[Fase 10: Explicabilidad e Interpretabilidad](./Phase_10_Explicabilidad_Interpretabilidad.md)**: Validación visual mediante Grad-CAM e importancia clínica con SHAP.

### BLOQUE 5: Conclusiones, Impacto y Entrega
*   **[Fase 11: Conclusiones e Impacto](./Phase_11_Conclusiones_Impacto.md)**: Resumen de aportaciones, limitaciones detectadas y visión de futuro.
*   **[Fase 12: Manual Técnico](./Phase_12_Manual_Tecnico_Reproducibilidad.md)**: Guía de despliegue, estructura de código y reproducibilidad.

---

## 🚀 Cómo completar el proyecto
Para finalizar la entrega, se sugiere:
1.  **Revisión Final**: Validar que las gráficas generadas en `/reports/figures/` coinciden con los resultados descritos.
2.  **Bibliografía**: Adjuntar el archivo de referencias en formato APA (presente en el repositorio global).
3.  **Anexos**: Incluir los logs de entrenamiento de `lightning_logs` como prueba de ejecución.

---
*Documentación generada para el Proyecto de Posgrado en IA & Machine Learning - 2026*
