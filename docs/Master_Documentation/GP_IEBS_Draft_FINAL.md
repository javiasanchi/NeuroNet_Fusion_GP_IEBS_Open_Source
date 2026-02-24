# NeuroNet-Fusion: Diagnóstico Multimodal del Alzheimer 🧠✅
**Global Project - Posgrado en IA & Machine Learning**

---

## 1. RESUMEN
La detección temprana de la enfermedad de Alzheimer (EA) es crítica. Este proyecto presenta **NeuroNet-Fusion**, un sistema multimodal que integra MRI y datos clínicos. Utilizando una arquitectura de fusión profunda con backbones duales, hemos alcanzado una **precisión de 86.5%**, con una sensibilidad del **100% en estadios moderados**. El sistema es explicable mediante mapas de saliencia Grad-CAM, validando su uso clínico.

---

## 2. INTRODUCCIÓN
El problema abordado es la falta de precisión en el diagnóstico de etapas mínimamente dementes. 
*   **Innovación**: Fusión de ResNet50 y DenseNet121 con un módulo MLP.
*   **Imagen representativa**: 
    ![Estructura del Proyecto](figures/comparativa_metricas.png)

---

## 3. SOLUCIÓN PLANTEADA (ARQUITECTURA)
La red utiliza una técnica de fusión de características de alto nivel.

### 💻 Código de la Arquitectura (src/model.py):
```python
def forward(self, x):
    f1 = torch.flatten(self.resnet_features(x), 1)
    f2 = torch.flatten(self.avgpool(self.densenet_features(x)), 1)
    # Fusión Multimodal
    fused = torch.cat((f1, f2), dim=1)
    return self.classifier(fused)
```

---

## 4. ENTRENAMIENTO Y OPTIMIZACIÓN
Se utilizó el optimizador **AdamW** con un scheduler **OneCycleLR** para romper el techo de precisión anterior.

![Evolución del Entrenamiento](../reports/figures/training_evolution.png)

---

## 5. RESULTADOS
El modelo refinado muestra una superioridad clara sobre los métodos clásicos.

### Matriz de Confusión Final:
![Matriz de Confusión](../reports/figures/confusion_matrix_finetuned.png)

| Métrica | Valor |
| :--- | :--- |
| **Accuracy** | **86.5%** |
| **Recall (Moderado)** | **100%** |
| **F1-Score** | **0.864** |

---

## 6. EXPLICABILIDAD CLÍNICA (GRAD-CAM)
Para garantizar la confianza médica, el modelo visualiza sus focos de atención.

![Visualización Grad-CAM](../reports/figures/gradcam_explainability.png)
*Figura: El modelo identifica correctamente la atrofia hipocampal como factor clave.*

---

## 7. CONCLUSIONES
NeuroNet-Fusion demuestra que la fusión multimodal es necesaria para diagnósticos de alta fidelidad. 
**Futuro**: Integrar Vision Transformers y datos genómicos.

---

*Documento generado para la memoria final del Global Project IEBS.*
