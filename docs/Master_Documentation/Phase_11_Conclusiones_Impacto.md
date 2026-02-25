# Fase 11: Conclusiones e Impacto del Proyecto

## 11.1 Resumen de Hallazgos
El proyecto **NeuroNet-Fusion** ha demostrado que la integración de imágenes de resonancia magnética con datos clínicos mediante arquitecturas de Deep Learning es la vía más robusta para el diagnóstico automático de Alzheimer. Se ha superado el "techo" de los modelos clásicos (70%), alcanzando un **86.5% de precisión**.

## 11.2 Aportes Principales
- **Metodología Multimodal**: Creación de un pipeline de fusión funcional y estable.
- **Evidencia Visual**: Provisión de mapas de calor para uso médico.
- **Estandarización**: Unificación de datasets heterogéneos (ADNI/OASIS) en un formato común.

## 11.3 Limitaciones
- **Variabilidad de Hardware**: El modelo es sensibles a la calidad del escáner MRI.
- **Datos Longitudinales**: El estudio actual es transversal; el seguimiento del mismo paciente a lo largo del tiempo podría mejorar la detección de patrones de progresión.

## 11.4 Líneas Futuras
- **Transfer Learning Avanzado**: Explorar el uso de Vision Transformers (ViT) entrenados específicamente en imágenes médicas (Med-ViT).
- **Integración Genómica**: Incluir perfiles de secuenciación de ADN para una medicina personalizada.
- **Despliegue**: Creación de una interfaz web (Webapp) para consulta médica remota.
