
import matplotlib.pyplot as plt
import os

output_dir = r"e:\MACHINE LEARNING\proyecto_global_IEBS\reports\figures"
os.makedirs(output_dir, exist_ok=True)

def save_code_as_image(code, title, filename):
    # Dynamic height based on code lines + small buffer for title
    num_lines = len(code.strip().split('\n'))
    fig_height = num_lines * 0.25 + 0.6
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('off')
    fig.patch.set_facecolor('#1e1e1e')
    
    # Position title and code with minimal padding
    plt.text(0.01, 0.96, title, color='#ffffff', fontsize=14, weight='bold', 
             transform=ax.transAxes, verticalalignment='top')
    plt.text(0.01, 0.01, code.strip(), color='#d4d4d4', fontsize=11, family='monospace', 
             transform=ax.transAxes, verticalalignment='bottom')
    
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300, facecolor='#1e1e1e', pad_inches=0.05)
    plt.close()

code_roc = """
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
plt.show()
"""

code_importance = """
import matplotlib.pyplot as plt
import xgboost as xgb

# Feature importance por 'gain' (contribución al reducir la impureza)
feature_names = ['MMSE', 'CDR', 'FAQ', 'ADAS11', 'EDUCYEARS', 'AGE', 
                 'APOE4', 'Hippocampus', 'Entorhinal', 'MidTemp',
                 'Ventricles', 'ABETA', 'TAU', 'pTAU']

importance = model.get_booster().get_score(importance_type='gain')

# Ordenar y graficar
sorted_idx = np.argsort(list(importance.values()))
plt.barh(np.array(feature_names)[sorted_idx], 
        np.array(list(importance.values()))[sorted_idx])
plt.xlabel("XGBoost Feature Importance (Gain)")
plt.title("Ranking de Biomarcadores")
plt.show()
"""

save_code_as_image(code_roc, "Código 12.4: Cálculo de Curvas ROC", "codigo_12_4_roc.jpg")
save_code_as_image(code_importance, "Código 12.5: Análisis de Importancia de Variables", "codigo_12_5_importance.jpg")

# --- CHAPTER 13 CODES ---

code_gradcam = """
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np

def generate_gradcam(model, image_tensor, target_class=None):
    # Capa objetivo: última capa convolucional de ResNet50
    target_layers = [model.resnet_features[-1][-1].conv3]
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
    
    rgb_image = image_tensor.squeeze().permute(1,2,0).numpy()
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    
    return cam_image, grayscale_cam
"""

code_shap_beeswarm = """
import shap
import matplotlib.pyplot as plt

# Explicador TreeSHAP para XGBoost
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Beeswarm plot para la clase AD
shap.summary_plot(
    shap_values[:, :, 2], # Clase AD (índice 2)
    X_test,
    feature_names=FEATURE_COLUMNS,
    plot_type='beeswarm',
    show=False
)
plt.title('SHAP Beeswarm: Factores determinantes del diagnóstico AD')
plt.tight_layout()
plt.savefig('reports/figures/shap_beeswarm_AD.png', dpi=150)
"""

code_shap_patient = """
def explain_patient(model, explainer, x_patient, feature_names):
    \"\"\"Genera una explicación SHAP individual tipo waterfall.\"\"\"
    shap_vals = explainer.shap_values(x_patient.reshape(1, -1))
    base_value = explainer.expected_value[2]
    
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[0][:, 2],
            base_values=base_value,
            data=x_patient,
            feature_names=feature_names
        )
    )
"""

code_streamlit_narrative = """
# En app_diagnostics.py — generación de narrativa clínica
top_features = sorted(zip(FEATURE_COLUMNS, shap_vals_patient),
                      key=lambda x: abs(x[1]), reverse=True)[:3]

narrative = f"Los factores determinantes para este diagnóstico son:\\n"
for feat, val in top_features:
    direction = "elevado" if val > 0 else "reducido"
    narrative += f"• {feat} {direction} (SHAP={val:.3f}): "
    narrative += get_clinical_explanation(feat, val)
"""

save_code_as_image(code_gradcam, "Código 13.2.1: Implementación de Grad-CAM", "codigo_13_2_1_gradcam.jpg")
save_code_as_image(code_shap_beeswarm, "Código 13.3.1: SHAP Beeswarm Plot", "codigo_13_3_1_shap_beeswarm.jpg")
save_code_as_image(code_shap_patient, "Código 13.3.2: Explicación Individual (Waterfall)", "codigo_13_3_2_shap_patient.jpg")
save_code_as_image(code_streamlit_narrative, "Código 13.4: Generación de Narrativa Clínica", "codigo_13_4_narrativa.jpg")

# --- CHAPTER 14 CODES ---

code_14_3_streamlit_config = """
import streamlit as st
import joblib

# Configuración global de la página
st.set_page_config(
    page_title="NeuroNet-Fusion | Diagnóstico Alzheimer",
    page_icon="🧠",
    layout="wide"
)

# Estilo dark para entorno clínico
st.markdown(\"\"\"
<style>
.block-container { padding-top:0rem !important; }
.stApp { background-color: #0E1117; color: white; }
</style>\"\"\", unsafe_allow_html=True)

@st.cache_resource
def load_model(path):
    return joblib.load(path)
"""

code_14_3_1_panel = """
# Diseño de 4 columnas para agrupación de biomarcadores
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.write("### 🧠 Cognitivo")
    bc_mmse = st.slider("MMSE (0–30)", 0, 30, 26)
    bc_cdr  = st.select_slider("CDR", [0, 0.5, 1, 2, 3], value=0.5)

with c2:
    st.write("### 👤 Demografía")
    age   = st.slider("Edad (años)", 50, 95, 73)
    apoe4 = st.radio("APOE4", [0, 1])

with c3:
    st.write("### 🩻 MRI (V/ICV)")
    hippo = st.number_input("Hipocampo", 0.001, 0.010, 0.0048)

with c4:
    st.write("### 🔬 LCR & Ventr.")
    tau   = st.number_input("TAU pg/mL", 0, 1000, 320)
"""

code_14_4_1_gauge = """
# Gauge Chart: Medidor de confianza de la IA
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=probs[class_idx] * 100,
    title={'text': f"<b>Confianza IA</b>"},
    gauge={
        'axis': {'range': [0, 100], 'ticksuffix': '%'},
        'bar': {'color': clrs[class_idx]},
        'threshold': {'line': {'color': 'white', 'width': 3}, 'value': 85}
    }
))
"""

code_14_4_2_radar = """
# Radar Plot: Perfil multivariante del paciente (ATN)
fig_radar = go.Figure(go.Scatterpolar(
    r=values_norm + [values_norm[0]],
    theta=categories + [categories[0]],
    fill='toself',
    line_color=clrs[class_idx]
))
"""

code_14_5_atn = """
# Lógica NIA-AA 2018 (Sistema ATN)
ami_stat = "A+" if abeta < 900 else "A-"
tau_stat = "T+" if tau   > 450 else "T-"
neuro_stat = "N+" if hippo < 0.0048 else "N-"

# Generación de dictamen narrativo
perfil_atn = ('patológico de alto riesgo' 
             if ami_stat=='A+' and tau_stat=='T+' 
             else 'estable / sin anomalías')
"""

code_14_6_bash = """
# Instalación y ejecución del sistema
pip install streamlit joblib plotly pandas numpy scikit-learn xgboost

# Ejecución de la aplicación interactiva
cd Analytical_Biomarker_Project/src
streamlit run app_diagnostics.py
"""

code_14_3_2_inference = """
# Vector de entrada para el modelo
X_input = np.array([[bc_mmse, bc_cdr, bc_faq, adas, age, apoe4,
                     educat, hippo, ento, midtemp, vent, abeta, tau, ptau]])

# Generación de predicción y probabilidades
probs     = model.predict_proba(X_input)[0]
class_idx = np.argmax(probs)

# Mapeo de etiqueta final
resultado = lbls[class_idx]
confianza = probs[class_idx]
"""

code_14_6_env = """
# Variables de entorno requeridas para el modelo
MODEL_PATH = "Analytical_Biomarker_Project/models/neuro_fusion_final_v1.joblib"
DATA_PATH  = "Analytical_Biomarker_Project/data/processed/biomarkers.csv"
"""

code_14_7_agent = """
# Agente Clínico (NLP Módulo 8) - GPT-4o-mini
client = OpenAI(api_key=openai_api_key)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Eres un neurólogo experto... siguiendo el marco ATN."},
        {"role": "user", "content": f"Analiza estos biomarcadores: {contexto_paciente}"}
    ],
    temperature=0.3
)

ai_report = response.choices[0].message.content
"""

save_code_as_image(code_14_3_streamlit_config, "Código 14.3: Configuración y CSS de Streamlit", "codigo_14_3_streamlit_config.jpg")
save_code_as_image(code_14_3_1_panel, "Código 14.3.1: Panel de Entrada Multicolumna", "codigo_14_3_1_panel.jpg")
save_code_as_image(code_14_3_2_inference, "Código 14.3.2: Motor de Inferencia (Inference Engine)", "codigo_14_3_2_inference.jpg")
save_code_as_image(code_14_4_1_gauge, "Código 14.4.1: Gráfico de Medidor de Confianza (Gauge)", "codigo_14_4_1_gauge.jpg")
save_code_as_image(code_14_4_2_radar, "Código 14.4.2: Gráfico Radar de Biomarcadores", "codigo_14_4_2_radar.jpg")
save_code_as_image(code_14_5_atn, "Código 14.5: Lógica de Diagnóstico ATN", "codigo_14_5_atn.jpg")
save_code_as_image(code_14_6_bash, "Código 14.6: Comandos de Instalación y Despliegue", "codigo_14_6_bash.jpg")
save_code_as_image(code_14_6_env, "Código 14.6: Configuración de Variables de Entorno", "codigo_14_6_env.jpg")
save_code_as_image(code_14_7_agent, "Código 14.7: Agente Clínico Inteligente (NLP Módulo 8)", "codigo_14_7_agent.jpg")

print("Code images generated successfully.")
