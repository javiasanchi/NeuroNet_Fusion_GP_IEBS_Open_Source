# Fase 8: Entrenamiento y Optimización

## 8.1 Estrategia de Entrenamiento
Para maximizar el rendimiento de **NeuroNet-Fusion**, se implementó un flujo de entrenamiento en dos etapas:
1. **Pre-entrenamiento**: Ajuste de las capas superiores (clasificador) con el backbone congelado durante 20 épocas.
2. **Fine-Tuning**: Descongelación gradual de las capas de la ResNet50 con una tasa de aprendizaje reducida (**1e-5**) para permitir que la red se especialice en características neurológicas.

## 8.2 Hiperparámetros Óptimos
- **Optimizador**: **AdamW** (mejor gestión del decaimiento de pesos).
- **Learning Rate**: Dinámico mediante un scheduler **OneCycleLR** con pico en `1e-4`.
- **Función de Pérdida**: `CrossEntropyLoss` con **Label Smoothing (0.1)** para manejar la incertidumbre en los casos fronterizos entre estadios.
- **Batch Size**: 64 (optimizado para GPU de 12GB+).

## 8.3 Monitorización
Se implementaron sistemas de **Early Stopping** basados en la pérdida de validación para evitar que el modelo memorice el dataset de entrenamiento. Se logró la convergencia óptima en la época 85.
