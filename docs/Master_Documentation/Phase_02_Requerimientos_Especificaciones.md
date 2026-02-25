# Fase 2: Especificación de Requerimientos

Este documento detalla los requerimientos funcionales y no funcionales necesarios para el desarrollo y despliegue del sistema **NeuroNet-Fusion**.

## 2.1 Requerimientos Funcionales (RF)

| ID | Nombre | Descripción |
| :--- | :--- | :--- |
| **RF-01** | Ingesta Multidatos | Capacidad de procesar CSVs clínicos e imágenes MRI (JPG/PNG). |
| **RF-02** | Estandarización de Datos | Unificación de pacientes bajo identificadores únicos (RID). |
| **RF-03** | Pipeline de Imagen | Normalización de intensidades y redimensionado a 224x224 píxeles. |
| **RF-04** | Aumento de Datos | Aplicación de transformaciones (rotación, flip) para robustecer el modelo. |
| **RF-05** | Clasificación Multiclase | Identificación de 4 estadios: Sano, Muy Leve, Leve y Moderado. |
| **RF-06** | Generación de Saliencia | Implementación de mapas Grad-CAM para visualización de atrofias. |

## 2.2 Requerimientos No Funcionales (RNF)

| ID | Nombre | Descripción |
| :--- | :--- | :--- |
| **RNF-01** | Portabilidad | Ejecución compatible con Windows 11 y soporte para GPUs NVIDIA (CUDA). |
| **RNF-02** | Escalabilidad | Arquitectura modular que permita cambiar el backbone (ej. de ResNet a ViT). |
| **RNF-03** | Rendimiento | Uso de precisión mixta (FP16) para optimizar el uso de VRAM. |
| **RNF-04** | Reproducibilidad | Entorno virtualizado y fijación de semillas aleatorias (seeds). |

## 2.3 Pila Tecnológica
- **Lenguaje**: Python 3.12+
- **Framework DL**: PyTorch 2.2+ & Torchvision.
- **Procesamiento**: Pandas, NumPy, OpenCV, Scikit-learn.
- **Hardware Recomendado**: GPU NVIDIA RTX 4070 o superior (min. 12GB VRAM).
