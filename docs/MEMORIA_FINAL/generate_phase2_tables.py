import matplotlib.pyplot as plt
import pandas as pd
import os

# Ensure directory exists
output_dir = r"e:\MACHINE LEARNING\proyecto_global_IEBS\Analytical_Biomarker_Project\reports\figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_table_as_image(df, title, filename, col_widths=None):
    fig, ax = plt.subplots(figsize=(12, len(df)*0.8 + 1.5))
    ax.axis('tight')
    ax.axis('off')
    
    # Header colors (NeuroNet blue)
    header_color = '#1E3A5F'
    cell_color = '#F8F9FA'
    alt_cell_color = '#E9ECEF'
    text_color = '#333333'
    header_text_color = '#FFFFFF'

    table = ax.table(cellText=df.values, 
                    colLabels=df.columns, 
                    cellLoc='left', 
                    loc='center',
                    colWidths=col_widths)
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)

    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header
            cell.set_text_props(weight='bold', color=header_text_color)
            cell.set_facecolor(header_color)
            cell.set_edgecolor('#DDDDDD')
        else:
            cell.set_text_props(color=text_color)
            cell.set_facecolor(cell_color if i % 2 == 0 else alt_cell_color)
            cell.set_edgecolor('#DDDDDD')
            if j == 0:
                 cell.get_text().set_weight('bold')

    plt.title(title, fontsize=16, pad=30, weight='bold', color='#1E3A5F')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# TABLE 2.1: EPIDEMIOLOGY
data21 = {
    "Indicador Epidemiológico": [
        "Prevalencia Global", "Tasa de Incidencia", "Costo Económico", 
        "Mortalidad", "Frecuencia por Edad", "Brecha Diagnóstica"
    ],
    "Impacto Actual (2024)": [
        "~55 millones de personas", "10 millones de casos nuevos / año", "1.3 billones de USD anuales",
        "7ª causa de muerte global (OMS)", "1 de cada 9 personas >65 años", "~75% de casos sin diagnóstico"
    ],
    "Proyección / Dato Crítico (2030-2050)": [
        "139 - 153 millones para 2050", "1 nuevo caso cada 3.2 segundos", "2.8 billones de USD para 2030",
        "Proyectada como 3ª causa global en 2040", "1 de cada 3 personas >85 años", "Incremento del 214% en países renta media-baja"
    ]
}
df21 = pd.DataFrame(data21)
save_table_as_image(df21, "Tabla 2.1: Impacto Epidemiológico Global Alzheimer (2024-2050)", "tabla_2_1_epidemiologia.png", [0.25, 0.35, 0.4])

# TABLE 2.2: CASCADE
data22 = {
    "Estadio Fisiopatológico": ["Fase Preclínica", "Fase Prodrómica", "Fase Demencia"],
    "Tiempo (vs. Síntoma)": ["-20 a -15 años", "-10 a -5 años", "Día 0 (Hoy)"],
    "Biomarcadores Alterados": ["↓ Aβ42 en LCR, ↑ Captación PET-Amiloide", "↑ pTau, Atrofia de Hipocampo y C. Entorrinal", "Atrofia cortical difusa, dilación ventricular"],
    "Estado Clínico": ["Asintomático (Normalidad aparente)", "Deterioro Cognitivo Leve (MCI)", "Alzheimer Establecido (AD)"]
}
df22 = pd.DataFrame(data22)
save_table_as_image(df22, "Tabla 2.2: Cascada Temporal y Ventana de Detección", "tabla_2_2_cascada.png")

# TABLE 2.3: GAP
data23 = {
    "Limitación Técnica": ["Unimodalidad", "Paradigma 2D", "Opacidad (Caja Negra)"],
    "Descripción": [
        "Modelos que solo analizan imagen (MRI) o solo datos (LCR/Clínica).",
        "Análisis de cortes aislados de MRI en lugar de volúmenes completos.",
        "Falta de herramientas de interpretación sobre la lógica del modelo."
    ],
    "Consecuencia Clínica (Impacto)": [
        "Diagnósticos incompletos al ignorar la correlación entre atrofia y biología.",
        "Pérdida de precisión en la detección de atrofia temprana del hipocampo.",
        "Desconfianza facultativa e incapacidad de validar la predicción legalmente."
    ]
}
df23 = pd.DataFrame(data23)
save_table_as_image(df23, "Tabla 2.3: Brecha Tecnológica en Sistemas de Diagnóstico IA", "tabla_2_3_brecha.png", [0.2, 0.4, 0.4])

# TABLE 2.4: INNOVATIONS
data24 = {
    "Pilar de Innovación": ["Fusión Multimodal", "Motor Volumétrico", "Interpretabilidad", "Agente Clínico"],
    "Implementación en NeuroNet-Fusion": [
        "Combinación de arquitecturas CNN (imagen) y MLP (biomarcadores).",
        "Procesamiento 3D (128³) que preserva la anatomía cerebral.",
        "Capas de Grad-CAM (Heatmaps) y SHAP (Ponderación de variables).",
        "Generación de narrativa médica mediante LLMs (GPT/Gemini)."
    ],
    "Valor Diferencial (KPI)": [
        "Aumento del +15% en Accuracy vs. modelos unimodales.",
        "Detección de atrofia en fases 10 años antes de la demencia.",
        "Validación visual de la zona de interés para el neurólogo.",
        "Reducción del 40% en tiempo de redacción de informes."
    ]
}
df24 = pd.DataFrame(data24)
save_table_as_image(df24, "Tabla 2.4: Innovaciones y KPIs de NeuroNet-Fusion", "tabla_2_4_innovaciones.png", [0.2, 0.4, 0.4])

print("Tablas generadas con éxito en reports/figures/")
