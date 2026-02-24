
import matplotlib.pyplot as plt
import pandas as pd
import os
import textwrap

# Create directory if it doesn't exist
output_dir = r"e:\MACHINE LEARNING\proyecto_global_IEBS\reports\figures"
os.makedirs(output_dir, exist_ok=True)

def save_table_as_image(df, title, filename, col_widths, title_pad=2, title_y=0.98):
    # Wrap text in all columns to prevent overflow - reduced width to 35 for narrow columns
    wrapped_df = df.copy()
    for col in wrapped_df.columns:
        wrapped_df[col] = wrapped_df[col].apply(lambda x: textwrap.fill(str(x), width=35) if isinstance(x, str) else x)

    # Factor de altura más ajustado para reducir espacio arriba/abajo
    fig_height = len(df) * 1.0 + 1.0
    fig, ax = plt.subplots(figsize=(14, fig_height)) 
    ax.axis('off')
    
    # Título pegado al borde superior de la tabla
    ax.set_title(title, fontsize=16, weight='bold', pad=title_pad, y=title_y)
    
    table = ax.table(
        cellText=wrapped_df.values, 
        colLabels=wrapped_df.columns, 
        loc='center', 
        cellLoc='center',
        colWidths=col_widths
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 4.0) # Ajustado para mantener legibilidad con menos aire
    
    # Style header and cells
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f8f9fa')
            else:
                cell.set_facecolor('#ffffff')

    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

# Table 12.1.1 - Values centered and optimized widths
data_12_1_1 = {
    "Métrica": ["Accuracy Global", "F1-Score (Weighted)", "AUC-ROC (Multiclase OvR)", "Kappa de Cohen"],
    "Valor": ["86.5%", "0.864", "0.898", "0.797"],
    "Interpretación Clínica": [
        "86 de cada 100 pacientes clasificados correctamente",
        "Excelente balance entre precisión y sensibilidad",
        "Muy alta capacidad discriminativa",
        "Acuerdo sustancial entre modelo y gold-standard"
    ]
}
df_12_1_1 = pd.DataFrame(data_12_1_1)
save_table_as_image(df_12_1_1, "Tabla 12.1.1: Resumen de Métricas Globales", "tabla_12_1_1_metricas.jpg", [0.25, 0.15, 0.60])

# Table 12.1.2
data_12_1_2 = {
    "Estadio Clínico": ["CN (Sano)", "MCI (Leve)", "AD (Alzheimer)", "AD Moderado (Mod)", "Accuracy Final"],
    "Precisión": ["0.87", "0.76", "0.82", "1.00", "-"],
    "Sensibilidad": ["0.74", "0.82", "0.89", "1.00", "-"],
    "F1-Score": ["0.80", "0.79", "0.85", "1.00", "0.865"],
    "Soporte": ["104", "95", "88", "113", "400"]
}
df_12_1_2 = pd.DataFrame(data_12_1_2)
save_table_as_image(df_12_1_2, "Tabla 12.1.2: Reporte de Clasificación Detallado", "tabla_12_1_2_clasificacion.jpg", [0.25, 0.15, 0.20, 0.20, 0.20])

# --- CHAPTER 13 TABLES ---

# Table 13.1 - Niveles de Explicabilidad
data_13_1 = {
    "Nivel": ["Producción", "Investigación"],
    "Técnica": ["SHAP (TreeSHAP)", "Grad-CAM"],
    "Modelo": ["XGBoost (14 Biomarcadores)", "CNN 2D (Benchmarking)"],
    "Rol": ["Explicabilidad del CDSS en uso", "Validación neuroanatómica"]
}
df_13_1 = pd.DataFrame(data_13_1)
save_table_as_image(df_13_1, "Tabla 13.1: Niveles de Explicabilidad e Interpretabilidad", "tabla_13_1_niveles.jpg", [0.20, 0.25, 0.25, 0.30])

# Table 13.3.1 - Interpretación SHAP
data_13_3_1 = {
    "Biomarcador": ["MMSE bajo", "CDR alto", "TAU alto", "Hipocampo bajo", "ABETA bajo", "APOE4=1", "Educación alta"],
    "Dirección": ["↑ P(AD)", "↑ P(AD)", "↑ P(AD)", "↑ P(AD)", "↑ P(AD)", "↑ P(AD)", "↓ P(AD)"],
    "Interpretación Clínica": [
        "Predictor más fuerte de EA establecida (MMSE < 20)",
        "Indicador de demencia clínicamente confirmada",
        "Refleja carga de ovillos neurofibrilares activa",
        "Atrofia hipocampal severa detectada",
        "Aβ42 < 900 = Secuestro en placas amiloides",
        "El alelo ε4 triplica el riesgo de EA esporádica",
        "Reserva cognitiva: actúa como factor protector"
    ]
}
df_13_3_1 = pd.DataFrame(data_13_3_1)
save_table_as_image(df_13_3_1, "Tabla 13.3.1: Interpretación de Impacto de Biomarcadores (SHAP)", "tabla_13_3_1_shap.jpg", [0.20, 0.15, 0.65], title_pad=5)

# Table 13.5 - Criterios SEN
data_13_5 = {
    "Criterio SEN": ["MMSE < 24 sugiere deterioro", "Hipocampo/ICV < 0.005 = atrofia", "TAU > 400 = Tau patológico", "APOE4 = factor de riesgo", "Educación = reserva cognitiva"],
    "Implementación": ["SHAP_MMSE > 0 si MMSE < 24", "Umbral N+ en sistema ATN", "Umbral T+ en sistema ATN", "Efecto positivo en P(AD)", "Efecto negativo en P(AD)"],
    "Concordancia": ["✅ 100%", "✅ 100%", "✅ 100%", "✅ 100%", "✅ 100%"]
}
df_13_5 = pd.DataFrame(data_13_5)
save_table_as_image(df_13_5, "Tabla 13.5: Consistencia con Criterios de la SEN", "tabla_13_5_sen.jpg", [0.35, 0.45, 0.20], title_pad=5, title_y=1.0)

# Table 13.3.2 - Ejemplo Paciente
data_13_patient = {
    "Biomarcador": ["MMSE", "CDR", "TAU", "ABETA-42", "Hipocampo/ICV", "APOE4", "EDUCYEARS"],
    "Valor": ["18 (Deterioro)", "2.0 (Moderado)", "580 (Alto)", "680 (Bajo)", "0.0031 (Atrofia)", "1 (Alelo ε4+)", "16 años"],
    "Impacto en Diagnóstico": ["Crítico (+)", "Sustancial (+)", "Sustancial (+)", "Moderado (+)", "Sustancial (+)", "Riesgo (+)", "Protector (-)"]
}
df_13_patient = pd.DataFrame(data_13_patient)
save_table_as_image(df_13_patient, "Tabla 13.3.2: Datos Clínicos de Paciente Ejemplo (AD)", "tabla_13_3_2_ejemplo_paciente.jpg", [0.25, 0.35, 0.40], title_pad=10, title_y=1.0)

# --- CHAPTER 14 TABLES ---

# Table 14.2 - Arquitectura de la Aplicación
data_14_2 = {
    "Módulo": ["1. CSS y Layout", "2. Carga del Modelo", "3. Panel de Entrada", "4. Motor de Inferencia", "5. Panel de Resultados", "6. Dictamen Clínico ATN"],
    "Descripción": [
        "Dark Theme optimizado para entorno clínico (Fullscreen).",
        "Carga asíncrona (joblib) con caché de recursos para rendimiento.",
        "Formulario de 4 columnas agrupando biomarcadores por categoría.",
        "Procesamiento de entrada y cálculo de probabilidad multiclase.",
        "Gráficos interactivos (Gauge Chart, Radar ATN, Barras).",
        "Lógica NIA-AA 2018 para informe diagnóstico estructurado."
    ]
}
df_14_2 = pd.DataFrame(data_14_2)
save_table_as_image(df_14_2, "Tabla 14.2: Módulos de la Arquitectura del Sistema", "tabla_14_2_arquitectura.jpg", [0.30, 0.70], title_pad=10, title_y=1.0)

# --- CLINICAL REPORT GENERATION ---

def save_report_as_image(report_text, filename):
    plt.figure(figsize=(10, 4.0), facecolor='#0E1117')
    plt.text(0.05, 0.95, report_text, family='monospace', fontsize=11, color='white', 
             verticalalignment='top', bbox=dict(boxstyle='round,pad=1', facecolor='#1A1F2E', alpha=1))
    plt.axis('off')
    plt.savefig(f"reports/figures/{filename}", dpi=300, bbox_inches='tight', facecolor='#0E1117')
    plt.close()

report_14_5 = """
NEURONET-FUSION | INFORME NEUROLÓGICO DIGITAL
FECHA: 23/02/2026 12:15
═══════════════════════════════════════════════════════════
PACIENTE: NF-PRO-73 | EDAD: 73 | APOE4: Portador

DIAGNÓSTICO IA: Alzheimer Establecido (AD)
CONFIANZA     : 91.4%
PERFIL ATN    : A+ / T+ / N+

COGNITIVO  : MMSE 22/30 | CDR 1.0 | FAQ 18/30
ESTRUCTURAL: Hippo 0.00350 | Ento 0.00310 | Vent 0.0580
MOLECULAR  : Abeta 680 pg/mL | TAU 540 pg/mL | pTAU 89 pg/mL
═══════════════════════════════════════════════════════════
Observación: Perfil patológico de alto riesgo. Atrofia 
hipocampal SEVERA. Biomarcadores moleculares consistentes 
con EA establecida. Se recomienda evaluación neurológica 
urgente y valoración para ensayo terapéutico anti-amiloide 
si elegible (MMSE 18-26).
═══════════════════════════════════════════════════════════
IEBS Business School — Proyecto Final Postgrado IA 2026
"""

save_report_as_image(report_14_5, "informe_ejemplo_final.jpg")

# --- CHAPTER 15 TABLES ---

# Table 15.2 - Consecución de Objetivos
data_15_2 = {
    "OE": ["OE-01", "OE-02", "OE-03", "OE-04", "OE-05", "OE-06", "OE-07", "OE-08", "OE-09"],
    "Objetivo Específico": [
        "Dataset maestro ADNI+OASIS-3",
        "Pipeline DICOM→NIfTI 3D",
        "Benchmarking 12 algoritmos",
        "Arquitectura fusión multimodal",
        "Optimización hiperparámetros",
        "Validación con métricas clínicas",
        "Explicabilidad Grad-CAM",
        "Importancia SHAP de biomarcadores",
        "Aplicación clínica Streamlit"
    ],
    "Estado": ["✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅"],
    "Resultado": [
        "11.606 sujetos procesados",
        "135 volúmenes; éxito 100%",
        "XGBoost seleccionado (86.5%)",
        "NeuroNet-Fusion Dual-Stream",
        "Optuna 100 trials; convergencia",
        "Recall 100% (AD), AUC 0.898",
        "Foco en hipocampo confirmado",
        "MMSE, CDR, Hippo = Top-3",
        "CDSS con Agente Clínico (NLP 4o-mini)"
    ]
}
df_15_2 = pd.DataFrame(data_15_2)
save_table_as_image(df_15_2, "Tabla 15.2: Estado de Consecución de Objetivos Específicos", "tabla_15_2_objetivos.jpg", [0.10, 0.40, 0.10, 0.40], title_pad=10, title_y=1.0)

# Table 15.4 - Limitaciones
data_15_4 = {
    "Limitación": ["Dataset de cohorte", "Validación clínica", "Análisis transversal", "Missing TAU/ABETA", "Generalización MRI"],
    "Descripción": [
        "ADNI no es representativa de la población hospitalaria real.",
        "No se ha realizado validación prospectiva real.",
        "Clasifica un punto temporal; no modela progresión.",
        "18.7% de valores faltantes en moleculares.",
        "Muestra 3D (135) aún pequeña para DL masivo."
    ],
    "Mitigación Futura": [
        "Validación con datos del SNS.",
        "Aprobación de Comité Ético (CE).",
        "Modelos temporales (LSTM/Transformer).",
        "Imputación probabilística (VAE).",
        "Escalar a los 11.606 volúmenes."
    ]
}
df_15_4 = pd.DataFrame(data_15_4)
save_table_as_image(df_15_4, "Tabla 15.4: Análisis de Limitaciones y Mitigación", "tabla_15_4_limitaciones.jpg", [0.25, 0.40, 0.35], title_pad=10, title_y=1.0)

# Final Reflection Banner
def save_reflection_banner(quote, author, filename):
    plt.figure(figsize=(10, 3), facecolor='#0E1117')
    plt.text(0.5, 0.7, f'"{quote}"', family='serif', style='italic', fontsize=18, color='#3498db', 
             ha='center', va='center', weight='bold')
    plt.text(0.5, 0.3, f"— {author} —", family='sans-serif', fontsize=12, color='white', 
             ha='center', va='center')
    plt.axis('off')
    plt.savefig(f"reports/figures/{filename}", dpi=300, bbox_inches='tight', facecolor='#0E1117')
    plt.close()

save_reflection_banner("La IA no reemplaza al neurólogo. Le da el tiempo que el paciente necesita.", 
                       "Conclusión NeuroNet-Fusion 2026", "banner_reflexion_final.jpg")

# Graph 12.5 - Feature Importance
import numpy as np

features = [
    'MMSE', 'CDR', 'FAQ', 'Hippocampus/ICV', 'TAU Total', 'ADAS-11', 
    'ABETA-42', 'Entorhinal/ICV', 'pTAU-181', 'AGE', 
    'Ventricles/ICV', 'APOE4', 'MidTemporal/ICV', 'EDUCYEARS'
]
importance_values = [
    1.000, 0.821, 0.742, 0.578, 0.431, 0.371, 
    0.312, 0.287, 0.261, 0.198, 0.174, 0.121, 0.108, 0.067
]

# Sort by importance
sorted_idx = np.argsort(importance_values)
features_sorted = [features[i] for i in sorted_idx]
values_sorted = [importance_values[i] for i in sorted_idx]

def save_feature_importance_graph(features, values, title, filename):
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(features, values, color='#3498db')
    
    # Add values on bars
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, str(v), color='#2c3e50', va='center', fontweight='bold')
    
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Importancia Relativa (Gain)', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#ffffff')
    
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

save_feature_importance_graph(features_sorted, values_sorted, "Ranking de Importancia de Biomarcadores (XGBoost)", "grafico_12_5_feature_importance.jpg")

# Executive Summary Results Table
data_resumen = {
    "Componente": [
        "Accuracy global (Test Set)", 
        "F1-Score ponderado", 
        "AUC-ROC multiclase", 
        "Kappa de Cohen", 
        "Sensibilidad — AD Moderado", 
        "Sensibilidad — Estadio MCI", 
        "Algoritmo final", 
        "Variables de entrada"
    ],
    "Resultado": [
        "86.5%", 
        "0.864", 
        "0.898", 
        "0.797", 
        "100%", 
        "82%", 
        "XGBoost (Optuna)", 
        "14 Biomarcadores (Tabular)"
    ]
}
df_resumen = pd.DataFrame(data_resumen)
save_table_as_image(df_resumen, "Tabla 0.1: Resumen de Resultados Clave de NeuroNet-Fusion", "tabla_resumen_ejecutivo.jpg", [0.45, 0.55])

# Table 3.2 - Objetivos Específicos
data_3_2 = {
    "ID": ["OE-01", "OE-02", "OE-03", "OE-04", "OE-05", "OE-06", "OE-07", "OE-08"],
    "Objetivo Específico": [
        "Adquirir y unificar datos de ADNI y OASIS-3",
        "Explorar procesamiento volumétrico MRI (Investigación)",
        "Benchmarking comparativo de 12 familias de algoritmos",
        "Diseñar y optimizar modelo de clasificación (Biomarcadores)",
        "Entrenamiento con optimización de hiperparámetros (Optuna)",
        "Validación con métricas clínicamente relevantes",
        "Analizar importancia de biomarcadores (SHAP)",
        "Desplegar sistema web de soporte clínico interactivo"
    ],
    "Herramienta / Método": [
        "Pandas, scripts Python",
        "dicom2nifti, nibabel",
        "Scikit-learn, XGBoost, PyTorch",
        "XGBoost + Optuna",
        "Optuna (100 trials, CV-5)",
        "Recall, F1, AUC, Kappa",
        "shap (TreeSHAP)",
        "Streamlit (Dashboard CDSS)"
    ],
    "Resultado Esperado": [
        "Dataset maestro de 11.606 sujetos",
        "135 volúmenes 3D procesados",
        "Selección empírica del algoritmo óptimo",
        "Accuracy >= 86% sobre 14 biomarkers",
        "Convergencia estable; AUC >= 0.89",
        "Sensibilidad >= 90% en AD moderado",
        "Ranking clínico de factores determinantes",
        "App CDSS con informe ATN"
    ]
}
df_3_2 = pd.DataFrame(data_3_2)
save_table_as_image(df_3_2, "Tabla 3.2: Objetivos Específicos del Proyecto NeuroNet-Fusion", "tabla_3_2_objetivos.jpg", [0.10, 0.35, 0.25, 0.30])

# Table 3.4 - Requerimientos No Funcionales
data_3_4 = {
    "ID": ["RNF-01", "RNF-02", "RNF-03", "RNF-04", "RNF-05"],
    "Requerimiento": ["Portabilidad", "Escalabilidad", "Rendimiento", "Reproducibilidad", "Seguridad Térmica"],
    "Especificación Técnica": [
        "Soporte para Windows 11 y Linux con integración CUDA 12.1.",
        "Diseño modular que permite intercambiar arquitecturas (XGBoost/ResNet/ViT).",
        "Sistema de entrenamiento en precisión mixta (FP16) para optimización de VRAM.",
        "Entornos aislados (venv) y fijación de semillas aleatorias (Random State 42).",
        "Implementación de ThermalThrottleCallback para protección de hardware."
    ]
}
df_3_4 = pd.DataFrame(data_3_4)
save_table_as_image(df_3_4, "Tabla 3.4: Requerimientos No Funcionales del Sistema", "tabla_3_4_rnf.jpg", [0.15, 0.25, 0.60])

# Table 3.6 - Core del Entorno (Pila Tecnológica)
data_3_6 = {
    "Categoría": [
        "Motor de Inferencia", 
        "Lenguaje Core", 
        "Machine Learning", 
        "Interpretabilidad", 
        "Interfaz Clínica", 
        "Procesamiento"
    ],
    "Software / Librería": [
        "joblib >= 1.3", 
        "Python 3.12", 
        "XGBoost, Scikit-learn, Optuna", 
        "SHAP (TreeSHAP)", 
        "Streamlit, Plotly", 
        "Pandas, NumPy, Matplotlib"
    ],
    "Motivo de Selección": [
        "Inferencia rápida <100ms.", 
        "Versatilidad y soporte médico.", 
        "Modelos SOTA para datos tabulares.", 
        "Validación clínica transparente.", 
        "Despliegue ágil de dashboards.", 
        "Estándar en Data Science."
    ]
}
df_3_6 = pd.DataFrame(data_3_6)
save_table_as_image(df_3_6, "Tabla 3.6: Pila Tecnológica Core del Sistema NeuroNet-Fusion", "tabla_3_6_pila_tecnologica.jpg", [0.25, 0.40, 0.35])

# Table 4.4 - Ventajas Competitivas de NeuroNet-Fusion
data_4_4 = {
    "Característica": [
        "Modalidades integradas", 
        "Número de biomarcadores", 
        "Marco diagnóstico", 
        "Explicabilidad", 
        "Interfaz clínica", 
        "Sensibilidad AD Moderado"
    ],
    "Modelos Previos": [
        "1-2", 
        "3-5", 
        "Empírico", 
        "Grad-CAM (Solo imagen)", 
        "Ninguna (Standalone)", 
        "85-90%"
    ],
    "NeuroNet-Fusion": [
        "3 (MRI + Clínico + Molecular)", 
        "14 (Multidominio)", 
        "ATN-NIA-AA 2018", 
        "Híbrida (Grad-CAM + SHAP)", 
        "Web (Streamlit + NLP)", 
        "100% (Gold Standard)"
    ]
}
df_4_4 = pd.DataFrame(data_4_4)
save_table_as_image(df_4_4, "Tabla 4.4: Ventajas Competitivas de NeuroNet-Fusion vs SOTA", "tabla_4_4_ventajas.jpg", [0.30, 0.35, 0.35])

# Table 6.1.2 - Inventario de Datos OASIS-3
data_6_1_2 = {
    "Categoría": [
        "Total de Sujetos", 
        "Rango de Edad", 
        "Seguimiento", 
        "Sesiones MRI", 
        "Biomarcadores LCR", 
        "Evaluación Clínica"
    ],
    "Valor": [
        "1,098", 
        "42 - 95 años", 
        "Hasta 30 años", 
        "2,100+", 
        "Aβ, Tau, pTau", 
        "MMSE, CDR, UDS v2/v3"
    ],
    "Descripción": [
        "Participantes con y sin deterioro cognitivo.", 
        "Desde adultos jóvenes hasta ancianos centenarios.", 
        "Uno de los seguimientos longitudinales más largos.", 
        "Protocolos T1, T2, FLAIR y DTI estandarizados.", 
        "Alineado con el marco de investigación ATN.", 
        "Batería neuropsicológica completa y validada."
    ]
}
df_6_1_2 = pd.DataFrame(data_6_1_2)
save_table_as_image(df_6_1_2, "Tabla 6.1.2: Inventario de Datos y Biomarcadores OASIS-3", "tabla_6_1_2_oasis3.jpg", [0.25, 0.25, 0.50])

# Table 6.2 - Estadísticas del Dataset Maestro
data_6_2 = {
    "Categoría / Clase": [
        "Total Sujetos (N)", 
        "Variables de entrada", 
        "Cobertura diagnóstica", 
        "Periodo de Recogida",
        "----------------",
        "🟢 Sano (CN)", 
        "🟡 Deterioro Leve (MCI)", 
        "🔴 Alzheimer (AD)",
        "TOTAL CLASIFICADO"
    ],
    "Valor": [
        "11,606", 
        "14 Biomarcadores", 
        "99.2%", 
        "1987 - 2024",
        "----------------",
        "3,922", 
        "3,761", 
        "3,923",
        "11,606"
    ],
    "Porcentaje / Detalle": [
        "Sujetos únicos (ADNI+OASIS)", 
        "Cognitivo + MRI + LCR", 
        "Confirmación DXSUM", 
        "Rango histórico unificado",
        "----------------",
        "33.8%", 
        "32.4%", 
        "33.8%",
        "100.0%"
    ]
}
df_6_2 = pd.DataFrame(data_6_2)
save_table_as_image(df_6_2, "Tabla 6.2: Inventario Final y Distribución de Clases del Dataset Maestro", "tabla_6_2_estadisticas_maestro.jpg", [0.40, 0.25, 0.35])

# Table 6.3 - Resultados del EDA (Estadísticas por Clase)
data_6_3 = {
    "Variable": [
        "MMSE", 
        "Edad (años)", 
        "Hipocampo/ICV", 
        "Entorrinal/ICV", 
        "TAU (pg/mL)", 
        "ABETA (pg/mL)"
    ],
    "CN (Media±SD)": [
        "29.1 ± 1.0", 
        "72.4 ± 6.8", 
        "0.00621 ± 0.0009", 
        "0.00551 ± 0.0008", 
        "242.1 ± 98.3", 
        "1,142 ± 287"
    ],
    "MCI (Media±SD)": [
        "26.4 ± 2.8", 
        "74.1 ± 7.3", 
        "0.00521 ± 0.0010", 
        "0.00441 ± 0.0009", 
        "368.5 ± 147.2", 
        "889.3 ± 312"
    ],
    "AD (Media±SD)": [
        "22.3 ± 4.5", 
        "74.8 ± 8.1", 
        "0.00371 ± 0.0011", 
        "0.00311 ± 0.0010", 
        "512.4 ± 189.6", 
        "631.2 ± 298"
    ],
    "p-valor": ["< 0.001", "< 0.001", "< 0.001", "< 0.001", "< 0.001", "< 0.001"]
}
df_6_3 = pd.DataFrame(data_6_3)
save_table_as_image(df_6_3, "Tabla 6.3: Estadísticas Descriptivas de Biomarcadores por Estadio Diagnóstico", "tabla_6_3_eda_stats.jpg", [0.20, 0.20, 0.20, 0.20, 0.20])

# Table 6.4 - Valores Faltantes en las Variables Tabulares
data_6_4 = {
    "Variable": [
        "MMSE", 
        "TAU / ABETA", 
        "Hippocampus/ICV", 
        "APOE4"
    ],
    "% Missing (Total)": [
        "2.3%", 
        "18.7%", 
        "5.1%", 
        "0.8%"
    ],
    "Estrategia de Imputación": [
        "Mediana por clase diagnóstica.", 
        "Mediana + flag binario 'biomarker_available'.", 
        "Mediana por clase y cohorte.", 
        "Moda (mayoría no portadores)."
    ]
}
df_6_4 = pd.DataFrame(data_6_4)
save_table_as_image(df_6_4, "Tabla 6.4: Análisis de Valores Faltantes y Estrategias de Imputación", "tabla_6_4_missing_values.jpg", [0.25, 0.25, 0.50])

# Table 8.4 - Correlación de features GLCM
data_8_4 = {
    "Feature GLCM": ["Homogeneidad", "Contraste", "Energía", "Correlación"],
    "Correlación con DX": ["+0.42", "-0.38", "+0.35", "-0.29"],
    "Interpretación Clínica": [
        "Mayor homogeneidad = menos textura = atrofia",
        "Menor contraste = pérdida de diferenciación cortical",
        "Textura más uniforme en cerebros atróficos",
        "Menor correlación espacial en tejido dañado"
    ]
}
df_8_4 = pd.DataFrame(data_8_4)
save_table_as_image(df_8_4, "Tabla 8.4: Correlación de descriptores de textura GLCM con el Diagnóstico", "tabla_8_4_glcm_corr.jpg", [0.25, 0.20, 0.55])

# Table 8.5 - Dominios ATN
data_8_5 = {
    "Dominio ATN": ["A — Amiloide", "T — Tau", "N — Neurodeg.", "Cognitivo", "Demográfico"],
    "Features": [
        "ABETA", 
        "TAU, PTAU", 
        "Hippocampus, Entorhinal, MidTemp, Ventricles", 
        "MMSE, CDR, FAQ, ADAS11", 
        "AGE, APOE4, PTEDUCAT"
    ],
    "Proceso patológico": [
        "Carga de β-amiloide en LCR",
        "Fosforilación de proteína tau",
        "Atrofia y daño neuronal",
        "Manifestación funcional",
        "Factores de riesgo validados"
    ]
}
df_8_5 = pd.DataFrame(data_8_5)
save_table_as_image(df_8_5, "Tabla 8.5: Mapeo de Características al Marco ATN-NIA-AA", "tabla_8_5_atn_mapping.jpg", [0.20, 0.40, 0.40])

# Table 9.2 - Algoritmos Evaluados
data_9_2 = {
    "Paradigma": ["Lineales / Kernel", "Ensambles de Árboles", "Deep Learning (Tabular)", "Deep Learning (3D Imagen)"],
    "Algoritmos": [
        "Logistic Regression, SVM (RBF)", 
        "Random Forest, XGBoost, LightGBM, CatBoost", 
        "TabNet, MLP (128-D Projector)", 
        "SimpleCNN3D, ResNet3D, DenseNet3D"
    ],
    "Uso en Proyecto": [
        "Baselines de baja complejidad.", 
        "Modelos de producción (Gradient Boosting).", 
        "Exploración de proyecciones latentes.", 
        "Benchmarking de modalidad volumétrica."
    ]
}
df_9_2 = pd.DataFrame(data_9_2)
save_table_as_image(df_9_2, "Tabla 9.2: Inventario de Algoritmos Evaluados en el Benchmark", "tabla_9_2_algoritmos.jpg", [0.25, 0.40, 0.35])

# Table 9.3 - Resultados Benchmark
data_9_3 = {
    "Algoritmo / Modelo": ["Logistic Regression", "MLP (Tabular)", "ResNet3D (135 cases)", "CatBoost", "LightGBM", "XGBoost (Optuna)"],
    "Accuracy": ["64.2%", "78.4%", "60.0%", "84.1%", "85.3%", "86.5%"],
    "F1-Score (W)": ["0.64", "0.78", "0.58", "0.84", "0.85", "0.864"],
    "Tiempo Ent.": ["~1s", "~45s", "~30min", "~12s", "~5s", "~34s"]
}
df_9_3 = pd.DataFrame(data_9_3)
save_table_as_image(df_9_3, "Tabla 9.3: Resultados Comparativos del Benchmarking Multimodal", "tabla_9_3_bench_results.jpg", [0.35, 0.20, 0.25, 0.25])

# Table 10.2.2 - Justificación Arquitectura
data_10_2_2 = {
    "Decisión Técnica": ["ResNet50 vs ResNet18", "Backbone Dual", "LayerNorm vs BatchNorm", "Dropout 0.5"],
    "Justificación": [
        "Superioridad en capturar atrofia cortical fina.",
        "Captura morfología global + micro-textura de tejido.",
        "Independencia del batch-size; mayor estabilidad.",
        "Fuerte regularización ante alta dimensionalidad (3072-D)."
    ],
    "Impacto en Benchmarking": ["+4.2% Acc.", "+5.8% F1-MCI", "Convergencia rápida", "Reducción de Overfitting"]
}
df_10_2_2 = pd.DataFrame(data_10_2_2)
save_table_as_image(df_10_2_2, "Tabla 10.2.2: Justificación de Decisiones de Arquitectura NeuroNet-Fusion", "tabla_10_2_2_justificacion.jpg", [0.30, 0.45, 0.25])

# Table 10.4.1 - Comparativa Ensemble
data_10_4_1 = {
    "Estrategia": ["Modelo Único (XGBoost)", "Ensemble (Votación Blanda)"],
    "Accuracy": ["86.48%", "86.52%"],
    "F1-Score MCI": ["0.79", "0.80"],
    "Desviación Pred.": ["Alta", "Mínima (Estable)"],
    "Veredicto": ["Sensible a ruido", "Robusto para CDSS"]
}
df_10_4_1 = pd.DataFrame(data_10_4_1)
save_table_as_image(df_10_4_1, "Tabla 10.4.1: Comparativa de Rendimiento - Modelo Único vs. Ensemble de Producción", "tabla_10_4_1_ensemble.jpg", [0.30, 0.15, 0.15, 0.20, 0.20])

# Table 7.2.1 - Inventario de Variables (Tabular)
data_7_2_1 = {
    "Variable": ["MMSE", "CDR", "FAQ", "ADAS-11", "AGE / EDU", "APOE4", "Biomarcadores MRI", "Biomarcadores LCR"],
    "Tipo": ["Cognitiva", "Ordinal", "Funcional", "Cognitiva", "Demográfica", "Genética", "Volumétrica (ICV)", "Molecular"],
    "Fuente": ["ADNI/OASIS", "ADNI", "ADNI", "ADNI", "Demografía", "Genética", "FreeSurfer", "LCR"],
    "Rango / Detalle": ["0–30 pts", "0 / 0.5 / 1 / 2", "0–30 pts", "0–70 pts", "50–95 años", "0 (No) / 1 (Sí)", "4 regiones (H, E, MT, V)", "Aβ, TAU, pTAU"]
}
df_7_2_1 = pd.DataFrame(data_7_2_1)
save_table_as_image(df_7_2_1, "Tabla 7.2.1: Inventario de Biomarcadores y Rangos Clínicos", "tabla_7_2_1_inventario.jpg", [0.20, 0.20, 0.20, 0.40])

print("Tables and feature importance graph generated successfully.")
