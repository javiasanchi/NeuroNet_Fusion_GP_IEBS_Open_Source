# Fase 7: Arquitectura del Sistema de Inteligencia (NeuroNet-Fusion 3D)

La arquitectura de **NeuroNet-Fusion v2** se ha evolucionado hacia un paradigma de procesamiento volumétrico profundo para maximizar la sensibilidad clínica.

## 7.1 Componentes de la Arquitectura 3D
1.  **Extractor de Características 3D (Vision Stream)**: 
    *   **Backbone**: **3D-ResNet18 / 3D-ResNet50**. Se utiliza una red convolucional 3D diseñada para procesar tensores de voxels.
    *   **Función**: Detectar patrones de atrofia volumétrica en el hipocampo y la corteza entorrinal.
2.  **Procesador de Biomarcadores (Clinical Stream)**:
    *   **Arquitectura**: Perceptrón Multicapa (MLP) de 3 capas.
    *   **Función**: Integrar variables como la edad, el género y las puntuaciones de tests cognitivos (MMSE) para proporcionar contexto al análisis de imagen.
3.  **Módulo de Fusión Multimodal**:
    *   **Técnica**: *Intermediate Fusion* mediante concatenación de vectores de características y capas de atención.
    *   **Salida**: Clasificación multiclase (CN, MCI, AD) balanceada mediante pesos de clase.

## 7.2 Optimización para Hardware (RTX 4070)
*   **VRAM Utilization**: Diseño optimizado para cargar bloques de 128x128x128 voxels con un *Batch Size* de 8-16, maximizando los 16GB de memoria de video.
*   **Precision Mixta (AMP)**: Uso de `torch.cuda.amp` para acelerar el cálculo manteniendo la estabilidad numérica.

## 7.3 Diagrama Lógico de Fusión 3D
```text
[MRI 3D Volume] -> [3D-ResNet] -> [Latent Vector Image] ----┐
                                                            (CONCAT + Attention) -> [Dense] -> [Diagnóstico]
[Biomarcadores] -> [MLP]      -> [Latent Vector Clinical] --┘
```
