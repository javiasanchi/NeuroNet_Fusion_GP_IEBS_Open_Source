# Fase 10: Explicabilidad e Interpretabilidad Clínica

Para que un modelo de IA sea útil en medicina, debe ser una "Caja de Cristal".

## 10.1 Grad-CAM (Visualización de Atención)
Implementamos Mapas de Saliencia mediante el algoritmo **Grad-CAM**.
- **Funcionamiento**: El modelo resalta en color rojo/naranja las regiones de la imagen que más influyeron en su decisión.
- **Validación Médica**: En pacientes clasificados con Alzheimer, el modelo focaliza su atención de manera consistente en la **región del hipocampo** y los **lóbulos temporales**, coincidiendo con la literatura médica sobre atrofia cerebral.

## 10.2 Importancia de Variables (SHAP)
En la rama tabular, utilizamos valores **SHAP** para entender qué biomarcadores clínicos pesan más:
1. **MMSE**: El factor determinante principal.
2. **Edad**: Un factor de riesgo multiplicador.
3. **ADAS-13**: Crucial para diferenciar entre estadios leves.

## 10.3 Conclusión de Interpretabilidad
La combinación de Grad-CAM (dónde mirar) y SHAP (qué datos importan) proporciona al clínico una "segunda opinión" razonada, aumentando la confianza en el sistema **NeuroNet-Fusion**.
