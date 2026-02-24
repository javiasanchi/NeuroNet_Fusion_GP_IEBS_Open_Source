# FASE 5 — MARCO TEÓRICO: IA MULTIMODAL Y NEUROLOGÍA COMPUTACIONAL

---

## 5.1 Fundamentos de la Enfermedad de Alzheimer

### 5.1.1 Fisiopatología de la Cascada Amiloide
La **hipótesis de la cascada amiloide** (Hardy & Higgins, 1992) es el modelo fisiopatológico predominante de la EA. Postula que la acumulación del péptido **β-Amiloide-42 (Aβ42)** en el parénquima cerebral en forma de **placas seniles** es el evento iniciador que desencadena:
1. Hiperfosforilación de la proteína TAU → formación de **ovillos neurofibrilares** (NFT).
2. Disfunción sináptica y activación microglial → neuroinflamación crónica.
3. Muerte neuronal → **atrofia cortical** (hipocampo, corteza entorrinal, corteza frontal).
4. Deterioro cognitivo progresivo observable clínicamente.

### 5.1.2 Biomarcadores del Marco ATN
El marco ATN-NIA-AA 2018 clasifica los biomarcadores en tres grupos según el proceso patológico que reflejan:

| Dominio | Biomarcador | Método | Valor Patológico |
|:---|:---|:---|:---|
| **A (Amiloide)** | Aβ42 en LCR | Punción lumbar | < 900 pg/mL |
| | PET amiloide | Neuroimagen funcional | SUVR > 1.4 |
| **T (Tau)** | pTau-181 en LCR | Punción lumbar | > 23 pg/mL |
| | Tau total en LCR | Punción lumbar | > 450 pg/mL |
| | PET tau | Neuroimagen funcional | Estadio Braak III-V |
| **N (Neurodeg.)** | Hipocampo/ICV | MRI estructural T1 | < 0.0048 |
| | FDG-PET | Neuroimagen metabólica | Hipometabolismo temporo-parietal |
| | Atrofia cortical | MRI estructural T1 | Grosor cortical < 2.8mm |

---

## 5.2 Fundamentos de Deep Learning para Neuroimagen

### 5.2.1 Redes Neuronales Convolucionales (CNN)
Las CNN explotan la **invariancia traslacional** de los patrones en imágenes: un hipocampo atrófico se reconoce independientemente de su posición en el corte axial. La operación de convolución discreta 2D es:

$$f_{salida}(x, y) = \sum_{i}\sum_{j} I(x+i, y+j) \cdot K(i, j)$$

donde $I$ es el mapa de activación de entrada y $K$ el kernel aprendido.

### 5.2.2 ResNet: Aprendizaje Residual
La arquitectura ResNet (He *et al.*, 2016) introduce las **skip connections** que permiten entrenar redes de hasta 152 capas sin degradación del gradiente:

$$\mathcal{F}(x) = H(x) - x \quad \Rightarrow \quad H(x) = \mathcal{F}(x) + x$$

El bloque residual aprende la **función residual** $\mathcal{F}(x)$ respecto a la identidad, mostrando que si la capa no añade valor, puede aproximar $\mathcal{F}(x) \approx 0$ fácilmente, convirtiendo la capa en una identidad.

### 5.2.3 Layer Normalization
A diferencia de BatchNorm (dependiente del tamaño del batch), **LayerNorm** normaliza a lo largo de las *features* de cada muestra individualmente, siendo superior para datos heterogéneos de múltiples modalidades:

$$\hat{x}_i = \frac{x_i - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} \cdot \gamma + \beta$$

---

## 5.3 XGBoost: Gradient Boosting Extremo

XGBoost (Chen & Guestrin, 2016) es el algoritmo de Gradient Boosting más eficiente disponible gracias a sus innovaciones:

**Objetivo regularizado:**
$$\mathcal{L}(\phi) = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)$$

donde $\Omega(f) = \gamma T + \frac{1}{2}\lambda ||w||^2$ penaliza la complejidad del árbol.

**Ventajas específicas para biomarcadores clínicos:**
- **Sparsity-aware:** Maneja NaN nativamente en TAU/ABETA.
- **Ganancia de split:** Un split óptimo minimiza la suma ponderada de residuos.
- **GPU acceleration:** `tree_method='hist'` con `device='cuda'` — 40x más rápido.

---

## 5.4 SHAP: SHapley Additive exPlanations

SHAP (Lundberg & Lee, 2017) proporciona explicaciones **aditivas y consistentes** basadas en la teoría de juegos cooperativos de Shapley. El valor SHAP de la característica $i$ para la predicción $f(x)$ es:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f_S(x_S \cup \{i\}) - f_S(x_S)]$$

En el contexto clínico, $\phi_i$ mide **cuánto desplaza** el valor del MMSE del paciente la predicción respecto al diagnóstico promedio de la población.

---

## 5.5 Grad-CAM: Gradient-weighted Class Activation Mapping

Grad-CAM (Selvaraju *et al.*, 2017) genera mapas de saliencia para CNNs sin modificar la arquitectura:

1. **Gradientes de retropropagación** desde la clase objetivo hasta la última capa convolucional.
2. **GAP de los gradientes** para obtener pesos de importancia por canal: $\alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A_{ij}^k}$
3. **Combinación ponderada con ReLU:** $L_{GradCAM}^c = ReLU\left(\sum_k \alpha_k^c A^k\right)$

El resultado es un **mapa de calor** superpuesto sobre la MRI que indica qué regiones cerebrales determinaron el diagnóstico.
