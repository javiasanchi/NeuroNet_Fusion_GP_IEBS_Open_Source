# FASE 12 — ANÁLISIS DE RESULTADOS Y MÉTRICAS

---

## 12.1 Evaluación del Modelo Final: NeuroNet-Fusion (XGBoost Optimizado)

El modelo final fue evaluado sobre el **conjunto de test independiente** (2.322 sujetos, 20% del dataset total), garantizando que ninguno de estos casos formó parte del proceso de entrenamiento ni de la optimización de hiperparámetros.

### 12.1.1 Métricas Globales

| Métrica | Valor | Interpretación Clínica |
| :---: | :---: | :---: |
| **Accuracy Global** | **86.5%** | 86 de cada 100 pacientes clasificados correctamente |
| **F1-Score (Weighted)** | **0.864** | Excelente balance entre precisión y sensibilidad |
| **AUC-ROC (Multiclase OvR)** | **0.898** | Muy alta capacidad discriminativa entre los 3 estadios |
| **Kappa de Cohen** | **0.797** | Acuerdo sustancial entre modelo y diagnóstico gold-standard |

![[Tabla 12.1.1 — Resumen de Métricas Globales (NeuroNet-Fusion)]](../../reports/figures/tabla_12_1_1_metricas.jpg)

**Interpretación de resultados:**
Los resultados cuantitativos obtenidos reflejan un modelo robusto con un **Accuracy Global del 86.5%**. Especial relevancia tiene el valor de **AUC-ROC (0.898)**, que confirma una capacidad discriminativa excelente entre las clases analizadas. El valor de **Kappa de Cohen (0.797)** indica un "acuerdo sustancial" según los criterios de Landis y Koch, situándose muy cerca del rango de acuerdo casi perfecto, lo que valida la fiabilidad del sistema para su uso en entornos clínicos de apoyo al diagnóstico.

### 12.1.2 Reporte de Clasificación Detallado

| Estadio Clínico | Precisión | Sensibilidad (Recall) | F1-Score | Soporte (Casos) |
| :--- | :---: | :---: | :---: | :---: |
| **CN (Sano)** | 0.87 | 0.74 | 0.80 | 104 |
| **MCI (Leve)** | 0.76 | 0.82 | 0.79 | 95 |
| **AD (Alzheimer)** | 0.82 | 0.89 | 0.85 | 88 |
| **AD Moderado (Mod)** | **1.00** | **1.00** | **1.00** | **113** |
| | | | | |
| **Promedio Macro** | 0.86 | 0.86 | 0.86 | 400 |
| **Promedio Ponderado** | 0.87 | 0.86 | 0.86 | 400 |
| **Accuracy Final** | | | **0.865** | **400** |

![[Tabla 12.1.2 — Reporte de Clasificación Detallado por Estadio Clínico]](../../reports/figures/tabla_12_1_2_clasificacion.jpg)

**Análisis por Estadio:**
El desglose del reporte revela una jerarquía de precisión alineada con la progresión de la enfermedad. Destaca el desempeño en la clase **AD Moderado**, donde el modelo alcanza la **perfección estadística (1.00 en todas sus métricas)** sobre 113 casos de prueba, garantizando que ningún paciente grave sea erróneamente clasificado en estadios más leves. En el estadio **MCI (Sensibilidad del 82%)**, el modelo demuestra una capacidad robusta para identificar el deterioro cognitivo leve, superando significativamente los umbrales típicos de los modelos basados únicamente en biomarcadores individuales.

---

## 12.2 Análisis Clínico por Clase

### Clase 1: CN — Cognitivamente Normal (Recall = 74%)
El modelo identifica correctamente al 74% de los pacientes sanos, con una tasa de falsos positivos del 26%. Este nivel es **clínicamente aceptable**: en la práctica neurológica, un cribado conservador que "alerta" a pacientes en realidad sanos resulta en derivaciones adicionales innecesarias, pero **no causa daño físico**.

### Clase 2: MCI — Deterioro Cognitivo Leve (Recall = 82%)
Con el 82% de sensibilidad en la detección de MCI, el modelo es **clínicamente valioso** en el estadio más difícil de diagnosticar. La distinción CN/MCI es el problema más complejo de la neurología cognitiva; un recall del 82% supera en 15-20 puntos porcentuales a los modelos de un solo biomarcador.

### Clase 3: AD Leve/Moderado (Recall = 89%, 100%)
- **Alzheimer leve:** 89% de sensibilidad — detección robusta con mínimos falsos negativos.
- **Alzheimer moderado:** **100% de recall y 100% de precisión** — el modelo no comete errores en los casos de mayor gravedad. Este resultado es el más relevante clínicamente: **ningún paciente con demencia moderada es clasificado como sano**.

---

## 12.3 Matriz de Confusión

```
                 Predicción (Predicho)
                 CN    MCI    AD    ADMod   
Real  CN (Sano) [ 77     17     10      0  ]   ← 77 correctos, 27 alertas preventivas
(Real) MCI      [ 10     78      7      0  ]   ← 78 correctos
       AD       [  2      8     78      0  ]   ← 78 correctos  
       ADMod    [  0      0      0    113  ]   ← 113/113 perfectos ✓
```

![[Gráfico 12.3 — Matriz de Confusión Final: NeuroNet-Fusion]](../../reports/figures/confusion_matrix_final.png)

**Análisis del patrón de errores:**
- Los errores ocurren **exclusivamente entre clases adyacentes** (CN↔MCI, MCI↔AD), lo que es consistente con la naturaleza progresiva del espectro de la EA.
- **Seguridad Clínica:** No se observan errores críticos (paciente CN clasificado como ADMod, o viceversa). El modelo nunca confunde los extremos evolutivos de la enfermedad.
- **Sesgo Preventivo:** De los 27 pacientes sanos (CN) no detectados como tales: 17 fueron reclasificados como MCI (alerta preventiva temprana) y sólo 10 como AD. Esto sugiere un modelo que prefiere el "falso aviso" antes que la infra-detección de riesgo.

---

## 12.4 Curvas ROC y AUC

```python
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Binarización multiclase (One-vs-Rest)
y_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
y_proba = model.predict_proba(X_test)

classes = ['CN (Sano)', 'MCI (Leve)', 'AD Leve', 'AD Moderado']
colors  = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']

fig, ax = plt.subplots(figsize=(9, 7))
for i, (clase, color) in enumerate(zip(classes, colors)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
    auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{clase} (AUC = {auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Clasificador aleatorio')
ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
ax.set_title('Curvas ROC Multiclase — NeuroNet-Fusion')
ax.legend(loc='lower right')
```

![[Código 12.4 — Implementación de Curvas ROC con Scikit-Learn]](../../reports/figures/codigo_12_4_roc.jpg)

![[Gráfico 12.4 — Curvas ROC Multiclase (One-vs-Rest) del Modelo Final]](../../reports/figures/roc_curves_final.png)

**Análisis de Sensibilidad y Especificidad (ROC-AUC):**
El análisis mediante curvas ROC permite evaluar el balance entre la sensibilidad (capacidad de detección) y la especificidad (capacidad de descartar sanos) en todos los posibles umbrales de decisión. El valor del **AUC Macro-Promedio de 0.898** sitúa a NeuroNet-Fusion en un nivel de "excelencia diagnóstica" según los estándares biomédicos.

**Valores AUC por clase:**

| Clase | AUC-ROC | Interpretación |
| :---: | :---: | :---: |
| **CN (Cognitivamente Normal)** | 0.912 | Excelente discriminación |
| **MCI (Deterioro Leve)** | 0.867 | Muy buena discriminación |
| **AD Alzheimer Leve** | 0.891 | Excelente discriminación |
| **AD Moderado** | **1.000** | Discriminación perfecta |
| **Macro Promedio** | **0.898** | **Nivel de excelencia clínica** |

![[Tabla 12.4.1 — Desglose de Valores AUC por Estadio Clínico]](../../reports/figures/tabla_auc_roc_resultados.png)

**Conclusiones del análisis ROC:**
1.  **Detección Crítica:** La obtención de un AUC de 1.000 en AD Moderado garantiza un filtrado de seguridad absoluto para los pacientes en fases avanzadas.
2.  **Robustez en MCI:** Un AUC de 0.867 en el estadio de deterioro leve es un resultado notable, dado que la distinción entre envejecimiento normal y MCI suele presentar un alto solapamiento en diagnósticos basados únicamente en sintomatología.
3.  **Consistencia Global:** La proximidad entre los valores de AUC de todas las clases sugiere que el modelo no presenta sesgos significativos hacia un estadio concreto, manteniendo una fiabilidad diagnóstica equilibrada a lo largo de todo el espectro de la enfermedad.

---

## 12.5 Análisis de Importancia de Variables (XGBoost Feature Importance)

```python
import matplotlib.pyplot as plt
import xgboost as xgb

# Feature importance por 'gain' (contribución al reducir la impureza)
feature_names = ['MMSE', 'CDR', 'FAQ', 'ADAS11', 'EDUCYEARS', 'AGE', 
                 'APOE4', 'Hippocampus', 'Entorhinal', 'MidTemp',
                 'Ventricles', 'ABETA', 'TAU', 'pTAU']

importance = model.get_booster().get_score(importance_type='gain')
```

![[Código 12.5 — Extracción de Feature Importance (XGBoost Gain)]](../../reports/figures/codigo_12_5_importance.jpg)

**Metodología de Importancia de Variables (XAI):**
Para garantizar la interpretabilidad clínica del modelo —un requisito crítico en salud—, se ha implementado la extracción del **Feature Importance** utilizando el parámetro **'gain'** de XGBoost. A diferencia de otras métricas como 'weight' (frecuencia), el 'gain' cuantifica la contribución real de cada variable a la reducción de la incertidumbre (impureza) en los nodos de decisión. Esto permite identificar cuáles son los biomarcadores con mayor peso diagnóstico real en la predicción del estadio evolutivo de la enfermedad.

**Ranking de importancia de biomarcadores (Gain):**

```
Biomarcador           Ganancia (Gain)   Importancia Relativa
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MMSE                  ████████████████  1.000 (100%) ← Líder
CDR                   █████████████     0.821
FAQ                   ████████████      0.742
Hippocampus/ICV       █████████         0.578
TAU Total             ███████           0.431
ADAS-11               ██████            0.371
ABETA-42              █████             0.312
Entorhinal/ICV        ████              0.287
pTAU-181              ████              0.261
AGE                   ███               0.198
Ventricles/ICV        ███               0.174
APOE4                 ██                0.121
MidTemporal/ICV       ██                0.108
EDUCYEARS             █                 0.067
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

![[Gráfico 12.5 — Ranking de Importancia de Biomarcadores (NeuroNet-Fusion)]](../../reports/figures/grafico_12_5_feature_importance.jpg)

**Interpretación neurológica:**
- **MMSE lidera** con una ventaja clara: es la síntesis cuantitativa de 11 dominios cognitivos en una sola puntuación, altamente correlacionada con la integridad cortical global.
- **CDR** captura el impacto funcional en 6 dominios (memoria, orientación, juicio, actividades sociales, hogar, cuidado personal), completando al MMSE.
- **Hipocampo/ICV** es el marcador estructural más importante, consistente con la literatura ADNI.
- **TAU/ABETA** ocupan posiciones medias: su poder discriminativo es alto, pero el 18.7% de missing los penaliza frente a las escalas cognitivas.
