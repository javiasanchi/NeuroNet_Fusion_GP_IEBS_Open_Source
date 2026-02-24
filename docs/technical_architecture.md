# Especificación de la Arquitectura Técnica: NeuroNet-Fusion

## 1. Visión General del Sistema
NeuroNet-Fusion utiliza una arquitectura de red neuronal multimodal diseñada para capitalizar tanto la información estructural de las imágenes cerebrales como el contexto fisiológico de los datos clínicos.

## 2. Diagrama de Flujo de Datos
1. **Ingesta**: Lectura de `data/raw/`.
2. **Procesamiento**:
   - `make_dataset.py` -> Limpieza tabular y pre-proceso de imagen.
   - `build_features.py` -> Extracción de rasgos clásicos y embeddings.
3. **Entrenamiento**:
   - Modelos Clásicos (`train_classical.py`).
   - Modelo de Fusión (`train_deep.py`).
4. **Interpretación**:
   - `gradcam.py` -> Saliencia de imagen.
   - `interpret_classical.py` -> Importancia SHAP.

## 3. Arquitectura del Modelo: Fusion-Net (MGCA)
El núcleo de la solución de Deep Learning es un modelo de **MGCA (Meta-Guided Cross-Attention)**:

### Rama A: Codificador de Imagen (ResNet18)
- **Input**: Imágenes 224x224x3.
- **Función**: Extrae características espaciales del tejido cerebral.
- **Output**: Vector de características de 512 dimensiones.

### Rama B: Codificador Tabular (MLP)
- **Input**: Variables clínicas (Edad, MMSE, Biomarcadores, etc.).
- **Función**: Proyecta datos clínicos a un espacio latente compatible con la imagen.
- **Output**: Vector latente de 128 dimensiones.

### Mecanismo de Fusión: Cross-Attention
- La información tabular actúa como "guía" para la atención visual.
- El modelo aprende a resaltar regiones de la imagen (ej. hipocampo) basándose en las puntuaciones cognitivas del paciente.

## 4. Estructura de Directorios
```bash
final_de_proyecto_de_posgrado/
├── data/
│   ├── raw/         # Datos originales (inmutables)
│   └── processed/   # Datos listos para entrenamiento
├── docs/            # Documentación formal
├── models/          # Pesos y transformadores guardados
├── reports/         # Informes de rendimiento y gráficas
└── src/             # Código fuente modular
```

## 5. Pila Tecnológica
- **Motor DL**: PyTorch & PyTorch Lightning.
- **Base de Datos**: Archivos CSV/Parquet gestionados con Pandas.
- **Visión**: OpenCV y Scikit-Image.
- **Explicabilidad**: SHAP y Grad-CAM.
