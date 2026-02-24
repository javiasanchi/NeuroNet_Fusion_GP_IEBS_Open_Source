# Documentación Técnica: Proyecto NeuroNet-Fusion

## 1. Introducción y Objetivo
El proyecto **NeuroNet-Fusion** tiene como objetivo el desarrollo de un modelo predictivo avanzado para la detección y progresión de la enfermedad de Alzheimer (AD), utilizando un enfoque multimodal que combina datos clínicos, cognitivos, genéticos, volumétricos (MRI) y biomarcadores de líquido cefalorraquídeo (CSF).

## 2. Tipos de Datos y Origen
Se han integrado datos de dos fuentes principales de referencia internacional:
- **ADNI (Alzheimer's Disease Neuroimaging Initiative)**: Proporciona perfiles detallados de pacientes a lo largo del tiempo, incluyendo neuroimagen procesada y biomarcadores CSF.
- **OASIS-3 (Open Access Series of Imaging Studies)**: Aporta datos longitudinales de neuroimagen y evaluaciones clínicas para validar la generalización del modelo.

### Categorías de Datos Utilizados:
- **Cognitivos/Funcionales**: MMSE (Mini-Mental State Exam), CDR (Clinical Dementia Rating), FAQ (Functional Activities Questionnaire).
- **Demográficos**: Género, Años de Educación, Edad de entrada al estudio.
- **Genéticos**: Presencia del alelo APOE4.
- **Estructurales (MRI)**: Volúmenes de regiones clave como Hipocampo, Corteza Entorrinal, Lóbulo Temporal Medio (MidTemp) y Ventrículos.
- **Biomarcadores CSF**: Niveles de Proteína Beta-Amiloide (ABETA), Proteína Tau y Tau Fosforilada (PTAU).

## 3. Proceso de Normalización
Para corregir la variabilidad anatómica natural entre individuos (basada en el tamaño de la cabeza), se ha implementado una **Normalización por Volumen Intracraneal (ICV)**.

### Metodología de Normalización:
1. **Identificación de la Métrica de Control**: Se localizó el campo `EICV` (Estimated Total Intracranial Volume) en el dataset `UCSDVOL.rda` de ADNI como proxy del tamaño total cerebral.
2. **Cálculo de Ratios**: Cada volumen regional se divide por el ICV del individuo.
   - `Hippo_Norm = Hippocampus / ICV`
   - `Ento_Norm = Entorhinal / ICV`
   - `MidTemp_Norm = MidTemp / ICV`
   - `Vent_Norm = Ventricles / ICV`
3. **Escalamiento**: Esto permite que el modelo compare la atrofia cerebral de manera relativa, eliminando el sesgo de que cabezas más grandes tengan naturalmente órganos más grandes.

## 4. Benchmarking y Ajuste de Algoritmos
Se realizó una "Batalla de Algoritmos" utilizando el 30% de los registros (3,607 muestras) para test, evaluando el impacto de las nuevas variables normalizadas.

### Modelos Evaluados:
- **CatBoost**: Resultó ser el modelo líder inicial (Accuracy: 88.19%). Maneja eficientemente valores nulos y relaciones no lineales.
- **XGBoost (Optimizado)**: Tras un proceso de GridSearch, alcanzó un sólido **87.69%** (Accuracy) y un excelente desempeño en la clase AD.
- **RandomForest**: Buen desempeño base (86.55%).
- **LightGBM**: Desempeño inferior en esta configuración (82.42%).

### Estructura de Hiperparámetros (XGBoost Final):
Se seleccionó XGBoost por su balance entre velocidad de inferencia y precisión tras la optimización:
- **n_estimators**: 800
- **learning_rate**: 0.05
- **max_depth**: 6
- **subsample**: 0.8
- **colsample_bytree**: 0.8
- **Objective**: multi:softprob

## 5. Análisis de Importancia de Variables
El análisis reveló hallazgos cruciales sobre la arquitectura del cerebro:
- **BCCDR (73.5%)**: Es la variable con mayor peso predictivo, lo cual es coherente clínicamente ya que mide el estado funcional.
- **MidTemp_Norm (5.40%)**: Esta nueva variable normalizada se posicionó como la **segunda más importante** a nivel global, superando incluso al MMSE tradicional.
- **Ento_Norm (3.83%)**: Cuarta variable en importancia, confirmando que la normalización de la corteza entorrinal es un biomarcador de alta fidelidad.
- **Hippo_Norm (1.29%)** y **Vent_Norm (0.63%)**: Contribuyen significativamente al ajuste fino del diagnóstico.

## 6. Conclusiones y Selección de Modelo
La integración de la normalización por ICV y la inclusión de medidas del lóbulo temporal medio han mejorado notablemente la capacidad discriminativa del modelo. 

**Selección Final:** Se ha optado por el modelo **XGBoost Optimizado** debido a su consistencia y la claridad en la importancia de las variables, logrando un **88% de precisión ponderada** en el conjunto de test (30% de los datos).

Las variables normalizadas `MidTemp_Norm` y `Ento_Norm` se han consolidado como pilares del diagnóstico estructural, ocupando el Top 5 de importancia junto a los test cognitivos.
