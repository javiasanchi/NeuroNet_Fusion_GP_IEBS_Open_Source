# Fase 6: Ingeniería de Características

En esta fase, extraemos información de alto nivel que será procesada por los modelos.

## 6.1 Características de Imagen (Deep Features)
Utilizamos el backbone de **ResNet50** como extractor de características global.
- **Capa Global Average Pooling (GAP)**: Transforma los mapas de activación de la última capa convolucional en un vector de **2048 elementos**.
- **Embeddings**: Estos vectores representan la "huella digital" morfológica del cerebro del paciente.

## 6.2 Características Clínicas (Clinical Embedding)
La red MLP procesa los datos clínicos para generar un espacio latente de **128 dimensiones**. Este embedding captura la predisposición genética y el estado cognitivo funcional.

## 6.3 Estrategia de Fusión
- **Concatenación**: Unión de los vectores de imagen (2048) y clínico (128).
- **Cuello de Botella (Bottleneck)**: Reducción de dimensionalidad a 512 unidades mediante una capa densa con **Layer Normalization**.
- **Justificación**: La normalización de capa es vital para equilibrar las magnitudes de los datos provenientes de fuentes tan distintas (imágenes vs números clínicos).
