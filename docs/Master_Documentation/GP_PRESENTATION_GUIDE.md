# Guía de Presentación Visual: NeuroNet-Fusion 🧠🚀

Este documento detalla la estructura de la presentación (Slides), las infografías recomendadas y la ubicación de las imágenes y códigos clave para la defensa del proyecto.

---

## 📊 Estructura de Diapositivas

### Diapositiva 1: Portada
*   **Título**: NeuroNet-Fusion: Diagnóstico Multimodal Temprano del Alzheimer.
*   **Subtítulo**: Alineación Cruzada de Bioimagen y Datos Clínicos mediante Deep Learning.
*   **Visual**: Imagen de un cerebro con conexiones neuronales digitales.

### Diapositiva 2: El Problema (Contexto)
*   **Contenido**: El reto de la detección en la "zona gris" (Deterioro Cognitivo Leve).
*   **Infografía recomendada**: Gráfico de embudo que muestra cómo se pierden casos en el diagnóstico tradicional.
*   **Imagen**: `docs/Master_Documentation/Phase_01_Definicion_Problema.md` (Contexto médico).

### Diapositiva 3: La Solución (Arquitectura)
*   **Contenido**: Dual-Backbone (ResNet50 + DenseNet121) con Fusión Profunda.
*   **Código clave**:
    ```python
    self.classifier = nn.Sequential(
        nn.Linear(2048 + 1024, 512),
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )
    ```
*   **Visual**: Diagrama de bloques de las dos ramas convergiendo en la capa de fusión.

### Diapositiva 4: Metodología SOTA
*   **Contenido**: Preprocesamiento CLAHE + Optimizador AdamW + OneCycleLR.
*   **Imagen**: `reports/figures/training_evolution.png` (Curva de aprendizaje).
*   **Texto**: "Cómo rompimos el techo del 70% de precisión".

### Diapositiva 5: Resultados (Métricas de Élite)
*   **Contenido**: **86.5% Accuracy**, 100% Sensibilidad en casos moderados.
*   **Imagen**: `reports/figures/confusion_matrix_finetuned.png`
*   **Visual**: Tabla comparativa resaltando la superioridad sobre SVM y LR.

### Diapositiva 6: Explicabilidad (Caja de Cristal)
*   **Contenido**: Validación mediante Grad-CAM.
*   **Imagen**: `reports/figures/gradcam_explainability.png`
*   **Texto**: "Vemos lo que el modelo ve: Atención focalizada en el hipocampo".

### Diapositiva 7: Conclusiones y Futuro
*   **Contenido**: Escalabilidad a Vision Transformers (ViT) e integración genómica.
*   **Infografía**: Timeline de evolución del proyecto.

---

## 🖼️ Guía de Imágenes en la Memoria (Donde colocarlas)

| Sección Memoria | Imagen Sugerida | Ruta de Archivo |
| :--- | :--- | :--- |
| **Introducción** | Diagrama de Flujo del Proyecto | `reports/figures/comparativa_metricas.png` |
| **Desarrollo** | Evolución del Entrenamiento | `reports/figures/training_evolution.png` |
| **Resultados** | Matriz de Confusión | `reports/figures/confusion_matrix_finetuned.png` |
| **Evaluación** | Mapas Grad-CAM | `reports/figures/gradcam_explainability.png` |

---

## 💻 Integración de Código Maestro

Se recomienda incluir estos fragmentos en el **Anexo Técnico**:

1.  **Lógica de Fusión** (`src/model.py`): Muestra cómo se concatenan las características de ResNet y DenseNet.
2.  **Ciclo de Entrenamiento** (`src/live_train.py`): Muestra el uso de `OneCycleLR` y `LabelSmoothing`.

---
*Este documento es una guía para la creación de materiales visuales impactantes.*
