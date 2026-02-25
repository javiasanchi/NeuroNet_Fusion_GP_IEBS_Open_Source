# Fase 12: Manual Técnico y Guía de Reproducción

Este documento sirve como manual para desarrolladores e investigadores que deseen replicar o extender los resultados.

## 12.1 Estructura del Código
- `src/data/`: Scripts para la limpieza y preparación de datos.
- `src/models/`: Definición de la arquitectura `NeuroNetFusion` y loops de entrenamiento.
- `src/features/`: Lógica de extracción de texturas y embeddings.
- `src/reports/`: Scripts para generación de gráficas y reportes automáticos.

## 12.2 Instrucciones de Uso
1. **Instalación**: `pip install -r requirements.txt`.
2. **Preparación**: Colocar datos en `data/raw/` y ejecutar `python src/data/make_dataset.py`.
3. **Entrenamiento**: `python src/models/train_deep.py`.
4. **Evaluación**: `python src/models/evaluate_deep.py`.

## 12.3 Configuración del Entorno
Se recomienda el uso de un entorno virtual (`venv` o `conda`).
- **RAM**: Mínimo 16GB.
- **VRAM**: Mínimo 8GB para inferencia, 12GB para entrenamiento.
- **S.O.**: Windows 10/11 o Linux Ubuntu 22.04.

## 12.4 Persistencia de Modelos
Los pesos del modelo entrenado se guardan automáticamente en la carpeta `models/checkpoints/` en formato `.pth`, listos para ser cargados para inferencia.
