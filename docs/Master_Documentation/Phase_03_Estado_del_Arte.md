# Fase 3: Estado del Arte y Propuesta de Solución

## 3.1 Tendencias Actuales (2024-2026)
El diagnóstico de Alzheimer mediante IA ha evolucionado de modelos puramente visuales a sistemas **multimodales**.
- **Modelos de Imagen**: El uso de redes convolucionales (CNN) como ResNet y EfficientNet sigue siendo el estándar por su eficiencia.
- **Fusión de Datos**: Los mecanismos de **Cross-Attention** permiten que la red aprenda correlaciones entre la atrofia visual y los biomarcadores clínicos (ej. niveles de APOE4 o proteína Tau).
- **Explicabilidad**: Herramientas como SHAP y Grad-CAM son fundamentales para la adopción clínica, permitiendo a los médicos entender el "porqué" de una predicción.

## 3.2 Propuesta NeuroNet-Fusion
Nuestra propuesta se basa en una arquitectura de **Dobles Ramas (Dual-Stream)**:
1. **Rama de Visión**: Un extractor de características basado en **ResNet50** que analiza cortes sagitales y axiales de MRI.
2. **Rama Clínica**: Una red densa (MLP) que procesa variables sociodemográficas y cognitivas.
3. **Módulo de Fusión**: Integración de características mediante concatenación y normalización de capas (**LayerNorm**) para estabilizar el aprendizaje.

## 3.3 Ventajas Competitivas
- **Sensibilidad superior** en la detección de estadios medios y avanzados.
- **Robustez** frente a datos faltantes en la rama clínica.
- **Interpretabilidad** directa integrada en el pipeline de evaluación.
