# FASE 9 — BENCHMARKING DE ALGORITMOS

---

## 9.1 Motivación del Benchmarking

Antes de diseñar la arquitectura definitiva de NeuroNet-Fusion, se realizó un **benchmarking sistemático** de los principales paradigmas de aprendizaje automático disponibles para el problema de clasificación multiclase sobre biomarcadores clínicos. El objetivo era doble:

1. **Identificar el baseline de rendimiento** que la arquitectura final debía superar.
2. **Justificar empíricamente** la elección del algoritmo sobre el que se construyó el modelo de producción.

El benchmark se realizó sobre el **dataset tabular de biomarcadores** (14 variables, 11.606 pacientes) con validación cruzada estratificada de 5 folds.

---

## 9.2 Algoritmos Evaluados

Para asegurar una comparativa exhaustiva, se seleccionó un conjunto diverso de algoritmos que representan los tres paradigmas principales del aprendizaje automático actual. Esta selección permite evaluar si la complejidad adicional de modelos profundos compensa frente a la eficiencia de los métodos de ensamble en datos tabulares clínicos.

1. **Modelos Lineales y Basados en Kernel:** Actúan como baselines de baja complejidad (Logistic Regression) y alta capacidad de margen (SVM).
2. **Ensambles de Árboles:** Desde arquitecturas clásicas como Random Forest hasta los modernos Gradient Boosting optimizados (XGBoost, LightGBM, CatBoost), conocidos por su dominio en datos tabulares.
3. **Deep Learning (1D y 3D):** Incluye Perceptrones Multicapa (MLP) para datos tabulares y redes convolucionales (ResNet3D) para la extracción directa de características desde los volúmenes de imagen.

![[Tabla 9.2 — Inventario de Algoritmos Evaluados en el Benchmark]](../../reports/figures/tabla_9_2_algoritmos.jpg)

---

## 9.3 Resultados y Análisis del Benchmark

El pipeline de evaluación (Fig. 9.1) permitió obtener una visión clara de la capacidad de generalización de cada arquitectura. Los resultados fueron calculados sobre el 20 % del dataset reservado para test, asegurando que ningún rastro del proceso de optimización (Optuna) se filtrara en estas métricas.

![Código 9.3 — Pipeline de Benchmarking Comparativo](../../reports/figures/codigo_9_3_benchmark_pipeline.png)

### 9.3.1 Desempeño por Paradigma

Tras ejecutar las pruebas, observamos tres niveles de rendimiento claramente diferenciados:

1.  **Zona de Infra-ajuste (LogReg, RF, CNN3D-Simple):** Los modelos lineales y las CNNs con volúmenes completos pero pocos datos (135 casos 3D) se estancaron en el rango del **44-66 % de accuracy**. Esto indica que la relación entre los biomarcadores tabulares es no-lineal y compleja, mientras que los datos 3D son insuficientes para entrenar redes convolucionales desde cero.
2.  **Zona Intermedia (MLP, SVM, ResNet3D):** Los modelos con mayor capacidad de representación alcanzaron el **52-60 %**. Es notable que la ResNet3D, aun con pocos datos, logró un 60 %, sugiriendo que hay información volumétrica valiosa, pero difícil de capturar sin un dataset de miles de pacientes.
3.  **Zona de Excelencia (XGBoost, LightGBM, CatBoost):** Los algoritmos de Gradient Boosting dominaron el benchmark con resultados superiores al **81 %**. XGBoost se alzó como el campeón con un **86.5 % de accuracy** y un **AUC de 0.898**.

![[Tabla 9.3 — Resultados Comparativos del Benchmarking Multimodal]](../../reports/figures/tabla_9_3_bench_results.jpg)

> **Interpretación Clínica:** La brecha de rendimiento entre los modelos tabulares (86.5 %) y los de imagen pura (60 %) confirma que, para el diagnóstico triclase (CN/MCI/AD), los biomarcadores moleculares del LCR (TAU, ABETA) y las pruebas cognitivas (MMSE, ADAS11) tienen una densidad de información predictiva mucho mayor que la morfología cerebral capturada por la MRI bruta.

---

## 9.4 Análisis de Eficiencia y Selección del Modelo Champion

### 9.4.1 Justificación Técnica de XGBoost

La elección de **XGBoost** como el motor de predicción final no se basó únicamente en el accuracy, sino en un balance de factores de ingeniería y clínicos:

*   **Eficiencia Computacional:** XGBoost alcanzó su pico de rendimiento en 34 segundos sobre GPU. En comparación, la ResNet3D requirió más de 30 minutos (1.840s) para obtener un resultado 26 puntos inferior. Esta eficiencia permite realizar ciclos de re-entrenamiento frecuentes a medida que lleguen nuevos datos clínicos.
*   **Manejo de Errores de Muestra:** A diferencia de SVM o Redes Neuronales, XGBoost es inmune a la falta de escalado (StandardScaler no es estrictamente necesario) y maneja de forma nativa los valores faltantes (`NaN`), algo crucial dado que el 11 % de los registros clínicos carecían de algún biomarcador del LCR.
*   **Ranking de Importancia:** La capacidad nativa de proporcionar la importancia de las características facilitó la validación clínica, confirmando que las variables MMSE y CDR eran los principales motores del modelo, en concordancia con la práctica médica.

### 9.4.2 Configuración de Hiperparámetros (Best Trial)

Tras 100 iteraciones con el motor de búsqueda Bayesiana **TPESampler** de Optuna, se identificó la configuración óptima que minimiza el sobreajuste sin sacrificar exactitud:

![Código 9.4 — Hiperparámetros Óptimos XGBoost](../../reports/figures/codigo_9_4_best_params.png)


---

## 9.5 Comparativa Visual: Benchmark de Accuracy

![Gráfica comparativa de Accuracy — Benchmark de Algoritmos](../../reports/figures/grafica_benchmark_accuracy.png)

*Figura: Comparativa de accuracy de todos los algoritmos evaluados en el benchmark. XGBoost domina claramente sobre todos los paradigmas, incluyendo los modelos de Deep Learning sobre imagen 3D.*
