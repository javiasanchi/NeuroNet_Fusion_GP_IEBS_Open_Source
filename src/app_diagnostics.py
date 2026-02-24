import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import re
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

# Cargar variables de entorno (API Keys, Paths)
load_dotenv()

# Configuración ultra-ancha
st.set_page_config(page_title="NeuroNet-Fusion | AI Clinical Scanner", layout="wide", page_icon="🧠")

# Path del modelo optimizado para el despliegue (Linux/Docker & Local)
MODEL_PATH = os.getenv('MODEL_PATH', 'models/neuro_fusion_final_v1.joblib')

@st.cache_resource
def load_assets():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

data_pkg = load_assets()

# CSS Ultra-Compacto para vista en una pantalla
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .block-container { padding-top: 0.3rem !important; padding-bottom: 0 !important; padding-left: 1rem !important; padding-right: 1rem !important; max-width: 100% !important; }
    h1 { margin: 0 !important; font-size: 1.45rem !important; font-weight: 700 !important; }
    h3 { font-size: 0.85rem !important; color: #58A6FF !important; margin: 4px 0 !important; text-transform: uppercase; letter-spacing: 0.04em; }
    .stApp { background-color: #0D1117; color: #E6EDF3; }
    .stSlider { padding-top: 0 !important; padding-bottom: 0 !important; margin-bottom: -6px !important; }
    .stSlider label { font-size: 0.75rem !important; margin-bottom: 0 !important; }
    div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label, div[data-testid="stRadio"] label { font-size: 0.75rem !important; margin-bottom: 0 !important; }
    div[data-testid="stNumberInput"] input { font-size: 0.8rem !important; padding: 3px 8px !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.15rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.82rem !important; padding: 6px 14px !important; }
    .result-box { border-radius: 8px; padding: 10px 14px; text-align: center; margin-top: 8px; }
    .diag-label { font-size: 0.72rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.06em; }
    .diag-value { font-size: 1.05rem; font-weight: 700; margin-top: 2px; }
    .medical-report { background:#161B22; padding:10px; border-radius:8px; border:1px solid #30363D; font-family:'Consolas',monospace; color:#C9D1D9; font-size:0.78rem; max-height: 280px; overflow-y: auto; }
    section[data-testid="stSidebar"] .block-container { padding-top: 0.5rem !important; }
    section[data-testid="stSidebar"] .stTextArea textarea { font-size: 0.75rem !important; }
    div.stButton > button { padding: 4px 10px !important; font-size: 0.8rem !important; }
    p { font-size: 0.82rem !important; margin: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- BARRA LATERAL COMPLETA ---
st.sidebar.markdown("## 🧠 NeuroNet-Fusion")
st.sidebar.caption("Diagnóstico Multimodal · Alzheimer · v2.0")
st.sidebar.markdown("---")

# AI Provider
st.sidebar.markdown("**🤖 Agente IA**")
ai_provider = st.sidebar.selectbox("Proveedor", ["OpenAI", "Google Gemini"], index=0, label_visibility="collapsed")
if ai_provider == "OpenAI":
    env_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_api_key = st.sidebar.text_input("API Key", value=env_api_key, type="password", placeholder="sk-... (OpenAI)")
    selected_model = st.sidebar.selectbox("Modelo OpenAI", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
else:
    env_gemini_key = os.getenv("GEMINI_API_KEY", "")
    gemini_api_key = st.sidebar.text_input("API Key", value=env_gemini_key, type="password", placeholder="AIza... (Gemini)")
    selected_model = st.sidebar.selectbox("Modelo Gemini", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-1.5-flash-8b"], index=0)

def extract_from_text(text):
    data = {}
    patterns = {
        'mmse': r'MMSE(?:(?:\s+de)?[:\s]+|(?:\s+score)?[:\s]+)(\d+)',
        'age': r'(\d+)\s+años|edad[:\s]+(\d+)',
        'abeta': r'ABETA(?:(?:\s+de)?[:\s]+|(?:\s+en)?[:\s]+)(\d+)',
        'tau': r'TAU(?:(?:\s+elevado\s+a)?[:\s]+|(?:\s+de)?[:\s]+)(\d+)',
        'ptau': r'PTAU(?:(?:\s+de)?[:\s]+)(\d+)',
        'hippo': r'Hipocampo(?:(?:\s+de)?[:\s]+|(?:\s+en)?[:\s]+)(0\.\d+)',
        'ento': r'Entorrinal(?:(?:\s+de)?[:\s]+)(0\.\d+)',
        'vent': r'Ventr[íi]culos(?:(?:\s+de)?[:\s]+)(0\.\d+)',
        'faq': r'FAQ(?:(?:\s+de)?[:\s]+)(\d+)',
        'apoe': r'APOE4(?:(?:\s+es)?[:\s]+|(?:\s+portador)?[:\s]+)(si|sí|positivo|portador)',
        'cdr': r'CDR(?:(?:\s+de)?[:\s]+)(\d\.?\d?)',
    }
    for var, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if var == 'age':
                val = match.group(1) if match.group(1) else match.group(2)
                data[var] = int(val)
            elif var == 'apoe':
                data[var] = 1
            elif var in ['hippo', 'ento', 'vent', 'cdr']:
                data[var] = float(match.group(1))
            else:
                data[var] = int(match.group(1))
    return data

st.sidebar.markdown("---")
st.sidebar.markdown("**📚 Glosario y Documentación**")
with st.sidebar:
    import streamlit.components.v1 as components
    components.html("""
    <button onclick="
        var tabs = window.parent.document.querySelectorAll('[data-baseweb=tab]');
        for (var i = 0; i < tabs.length; i++) {
            if (tabs[i].innerText.trim().includes('Documentaci')) {
                tabs[i].click();
                break;
            }
        }
    " style="
        width: 100%;
        background: linear-gradient(135deg, #1a3a6b, #1e4d8c);
        color: #58A6FF;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 9px 14px;
        font-size: 0.84rem;
        font-weight: 600;
        cursor: pointer;
        font-family: Inter, sans-serif;
        letter-spacing: 0.02em;
    " onmouseover="this.style.opacity='0.85'"
       onmouseout="this.style.opacity='1'">
        📚 Abrir Glosario y Documentación →
    </button>
    """, height=50)
st.sidebar.markdown("---")


# NLP Scanner en sidebar
st.sidebar.markdown("**🔍 Escáner NLP de Informes**")
raw_text_sidebar = st.sidebar.text_area("Pega el informe clínico:", height=90, placeholder="Ej: Varón 75 años, MMSE 22, TAU 480, Hipocampo 0.003...")
c_side1, c_side2 = st.sidebar.columns(2)
with c_side1:
    if st.button("🚀 Escanear", use_container_width=True):
        extracted = extract_from_text(raw_text_sidebar)
        if extracted:
            for k, v in extracted.items():
                st.session_state[k] = v
            st.sidebar.success(f"✅ {len(extracted)} biomarcadores.")
        else:
            st.sidebar.warning("No detectado.")
with c_side2:
    if st.button("🗑️ Limpiar", use_container_width=True):
        for k in ['mmse', 'age', 'abeta', 'tau', 'ptau', 'hippo', 'ento', 'vent', 'faq', 'apoe', 'cdr', 'p_name', 'p_id', 'p_history', 'p_physician']:
            st.session_state[k] = "" if 'p_' in k else 0
        st.rerun()

# Inicializar estado si no existe
defaults = {
    'mmse': 24, 'age': 72, 'abeta': 850, 'tau': 450, 'ptau': 65, 
    'hippo': 0.0045, 'ento': 0.0028, 'vent': 0.08, 'faq': 10, 'apoe': 0, 'cdr': 0.5,
    'p_name': "", 'p_id': "", 'p_history': "", 'p_physician': ""
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- NAVEGACIÓN PRINCIPAL ---
tab_app, tab_doc = st.tabs(["🧠 NeuroNet-Fusion AI", "📚 Documentación Técnica"])

with tab_app:
    # Header ultra-compacto
    st.markdown("""
    <div style='display:flex; align-items:center; justify-content:space-between; padding: 4px 0 6px 0; border-bottom: 1px solid #21262D; margin-bottom:8px;'>
        <div>
            <span style='font-size:1.3rem; font-weight:800; color:#E6EDF3;'>🧠 NeuroNet-Fusion</span>
            <span style='font-size:0.75rem; color:#58A6FF; margin-left:10px;'>Diagnóstico Multimodal · Alzheimer · v2.0</span>
        </div>
        <span style='background:#1e3a5f; color:#58A6FF; padding:3px 12px; border-radius:20px; font-size:0.73rem; border:1px solid #30363D;'>
            📚 Ver glosario → pestaña <b>Documentación Técnica</b>
        </span>
    </div>
    """, unsafe_allow_html=True)

    if data_pkg:
        model, features = data_pkg['model'], data_pkg['features']

        # NUEVO: Bloque de Datos Administrativos y Clínicos
        with st.expander("📋 1. Identificación y Antecedentes (Opcional para el Neurólogo)", expanded=False):
            c_adm1, c_adm2 = st.columns(2)
            with c_adm1:
                p_name = st.text_input("Nombre del Paciente", value=st.session_state.p_name, placeholder="Ej: Don Juan Martínez", key="p_name_inp")
                p_id = st.text_input("ID / Nº Historia Clínica", value=st.session_state.p_id, placeholder="Ej: HC-12345/2026", key="p_id_inp")
            with c_adm2:
                p_physician = st.text_input("Neurólogo Responsable", value=st.session_state.p_physician, placeholder="Dr/a. García", key="p_phys_inp")
                p_date = st.date_input("Fecha Clínica", format="DD/MM/YYYY")
            
            p_history = st.text_area("Motivo de Consulta y Antecedentes Médicos", value=st.session_state.p_history, placeholder="Ej: Pérdida persistente de memoria episódica, antecedentes de hipertensión...", key="p_hist_inp")
            
            # Persistir en session_state para el informe
            st.session_state.p_name = p_name
            st.session_state.p_id = p_id
            st.session_state.p_history = p_history
            st.session_state.p_physician = p_physician

        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

        # Layout 3 Columnas: Perfil | ATN | Resultado
        col1, col2, col3 = st.columns([1.1, 1.1, 0.9], gap="medium")

        with col1:
            st.markdown("### 👤 Perfil y Cognición")
            with st.container(border=True):
                c1, c2 = st.columns(2)
                with c1:
                    age = st.number_input("Edad", 50, 95, int(st.session_state.age))
                    educat = st.number_input("Educación (Años)", 0, 25, 12, key="educ_inp")
                with c2:
                    gender = st.selectbox("Género", ["Masculino", "Femenino"], index=1)
                    apoe4 = st.radio("APOE4", [0, 1], format_func=lambda x: "Portador" if x==1 else "No portador", horizontal=True, index=int(st.session_state.apoe))
                
                bc_mmse = st.slider("MMSE (Cognición)", 0, 30, int(st.session_state.mmse))
                bc_cdr = st.select_slider("CDR (Gravedad)", options=[0, 0.5, 1, 2], value=float(st.session_state.cdr))
                bc_faq = st.slider("FAQ (Funcionalidad)", 0, 30, int(st.session_state.faq))

        with col2:
            st.markdown("### 🧪 Imagen y LCR (ATN)")
            with st.container(border=True):
                hippo = st.slider("Hipocampo (Vol. Norm.)", 0.001, 0.010, float(st.session_state.hippo), step=0.0001, format="%.4f")
                ento = st.slider("Entorrinal (Vol. Norm.)", 0.001, 0.010, float(st.session_state.ento), step=0.0001, format="%.4f")
                vent = st.slider("Ventrículos (Vol. Norm.)", 0.010, 0.100, float(st.session_state.vent), format="%.4f")
                c3, c4, c5 = st.columns(3)
                with c3: abeta = st.number_input("Aβ42", 200, 2000, int(st.session_state.abeta))
                with c4: tau = st.number_input("Tau T.", 50, 1500, int(st.session_state.tau))
                with c5: ptau_val = st.number_input("pTau", 10, 200, int(st.session_state.ptau))

        # --- PREDICCIÓN en tiempo real (col3) ---
        gender_val = 1 if gender == "Femenino" else 0
        row_data = [bc_mmse, bc_cdr, bc_faq, gender_val, educat, age, apoe4, hippo, ento, 0.0185, vent, abeta, tau, ptau_val]
        input_data = pd.DataFrame([row_data], columns=features)
        probs = model.predict_proba(input_data)[0]
        class_idx = np.argmax(probs)
        lbls_short = ["CN", "MCI", "AD"]
        lbls = ["Cognitivamente Normal (CN)", "Deterioro Cognitivo Leve (MCI)", "Enfermedad de Alzheimer (AD)"]
        colors = ["#10b981", "#f59e0b", "#ef4444"]

        with col3:
            st.markdown("### 📊 Diagnóstico")
            with st.container(border=True):
                st.markdown(f"""
                <div style='background:{colors[class_idx]}18; border:2px solid {colors[class_idx]}; border-radius:10px; padding:10px 6px; text-align:center; margin-bottom:6px;'>
                    <div style='font-size:0.68rem; opacity:0.8; text-transform:uppercase; letter-spacing:0.08em;'>Predicción Principal</div>
                    <div style='font-size:1.5rem; font-weight:800; color:{colors[class_idx]};'>{lbls_short[class_idx]}</div>
                    <div style='font-size:0.73rem; color:{colors[class_idx]};'>{probs[class_idx]*100:.1f}% confianza</div>
                </div>
                """, unsafe_allow_html=True)
                # Gráfico Plotly compact
                fig = go.Figure(go.Bar(
                    x=probs * 100,
                    y=lbls_short,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{p*100:.1f}%" for p in probs],
                    textposition='auto',
                    textfont=dict(size=11)
                ))
                fig.update_layout(
                    height=110,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E6EDF3', size=11),
                    xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                if st.button("✨ Generar Informe IA", use_container_width=True, key="gen_btn"):
                    st.session_state['run_ai'] = True

        # --- GENERACIÓN DE INFORME IA ---
        if st.session_state.get('run_ai', False):
            st.session_state['run_ai'] = False
            current_key = openai_api_key if ai_provider == "OpenAI" else gemini_api_key
            if not current_key:
                st.warning(f"Introduce tu API Key de {ai_provider} en la barra lateral.")
            else:
                try:
                    with st.spinner("Redactando informe clínico..."):
                        # Construir contexto enriquecido con datos administrativos
                        identificacion = f"PACIENTE: {st.session_state.p_name or 'N/A'} | ID: {st.session_state.p_id or 'N/A'}\nFECHA: {p_date if 'p_date' in locals() else 'Hoy'}\nFIRMADO POR: {st.session_state.p_physician or 'Neurólogo de Guardia'}"
                        antecedentes = f"ANTECEDENTES/MOTIVO: {st.session_state.p_history or 'No proporcionado'}"
                        biomarcadores = f"DATOS BIOLÓGICOS:\n- Perfil: {age} años, {'Mujer' if gender_val==1 else 'Varón'}, {educat} años educ.\n- APOE4: {'Portador' if apoe4==1 else 'No portador'}\n- Cognición: MMSE {bc_mmse}/30, CDR {bc_cdr}, FAQ {bc_faq}/30\n- MRI: Hipocampo {hippo:.4f}, Entorrinal {ento:.4f}, Ventrículos {vent:.4f}\n- LCR: Aβ42 {abeta}, Tau {tau}, pTau {ptau_val}\n- Predicción NeuroNet: {lbls[class_idx]} ({probs[class_idx]*100:.1f}%)"
                        
                        contexto_paciente = f"{identificacion}\n\n{antecedentes}\n\n{biomarcadores}"
                        
                        system_prompt = "Eres un neurólogo experto. Genera un informe médico formal y estructurado. Debes incluir encabezados para: Datos del Paciente, Justificación Clínica, Resultados de Biomarcadores, Interpretación ATN y Recomendaciones. El tono debe ser de rigor académico y clínico."
                        if ai_provider == "OpenAI":
                            client = OpenAI(api_key=openai_api_key)
                            response = client.chat.completions.create(model=selected_model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": contexto_paciente}], temperature=0.3)
                            ai_report = response.choices[0].message.content
                        else:
                            genai.configure(api_key=gemini_api_key)
                            model_gem = genai.GenerativeModel(model_name=selected_model, system_instruction=system_prompt)
                            ai_report = model_gem.generate_content(contexto_paciente).text
                    st.markdown("#### 📄 Informe del Agente")
                    st.markdown(f"<div class='medical-report'>{ai_report}</div>", unsafe_allow_html=True)
                    st.download_button("📥 Descargar (.md)", ai_report, file_name=f"Informe_{age}.md", use_container_width=True)
                except Exception as e:
                    st.error(f"Error AI: {str(e)}")

    else:
        st.error("❌ Error: No se pudo cargar el modelo de diagnóstico.")

with tab_doc:
    # Botón de retorno a la app usando JS
    components.html("""
    <button onclick="
        var tabs = window.parent.document.querySelectorAll('[data-baseweb=tab]');
        if (tabs.length > 0) { tabs[0].click(); }
    " style="
        background: linear-gradient(135deg, #1a3a6b, #1e4d8c);
        color: #58A6FF;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 7px 18px;
        font-size: 0.82rem;
        font-weight: 600;
        cursor: pointer;
        font-family: Inter, sans-serif;
        letter-spacing: 0.02em;
    " onmouseover="this.style.opacity='0.82'"
       onmouseout="this.style.opacity='1'">
        ← Volver a la Aplicación
    </button>
    """, height=45)

    st.markdown("## 📚 Documentación Técnica — NeuroNet-Fusion")
    st.markdown("`Versión 2.0 | IEBS Digital School | Proyecto Fin de Máster | 2025`")
    st.markdown("---")

    # ── SECCIÓN 1: Qué es NeuroNet-Fusion ──────────────────────────────────
    with st.expander("🧠 ¿Qué es NeuroNet-Fusion?", expanded=True):
        st.markdown("""
**NeuroNet-Fusion** es un sistema de diagnóstico asistido por inteligencia artificial diseñado para apoyar a
profesionales médicos en la detección y estadificación de la **Enfermedad de Alzheimer (EA)**.

El sistema fusiona tres modalidades de datos clínicos — **cognitivos, estructurales (MRI) y moleculares (LCR)** —
para emitir un dictamen probabilístico multiclase:

| Clase | Descripción |
|:---|:---|
| **CN** — Cognitivamente Normal | Sin evidencia clínica ni biológica de patología |
| **MCI** — Deterioro Cognitivo Leve | Declive detectable, funcionalidad diaria preservada |
| **AD** — Alzheimer's Disease | Demencia tipo Alzheimer establecida |

**Arquitectura del Motor de Diagnóstico:**
El modelo utiliza **XGBoost (Extreme Gradient Boosting)**, entrenado sobre un dataset multicohorte obtenido de las iniciativas
internacionales **ADNI (Alzheimer's Disease Neuroimaging Initiative)** y **OASIS-3 (Open Access Series of Imaging Studies)**.
Combina 14 variables de entrada para producir probabilidades de pertenencia a las 3 clases.

**Base Normativa:** El marco de clasificación sigue las guías **NIA-AA 2018** (National Institute on Aging & Alzheimer's Association),
que establecen el sistema ATN como estándar de diagnóstico biológico.
        """)

    # ── SECCIÓN 2: Biomarcadores Cognitivos ────────────────────────────────
    with st.expander("🧩 Biomarcadores Cognitivos (MMSE · CDR · FAQ)"):
        st.markdown("""
### 1. MMSE — Mini-Mental State Examination
El **MMSE** es la escala de cribado cognitivo más utilizada en clínica. Evalúa:
- Orientación temporo-espacial
- Registro y memoria a corto plazo
- Atención y cálculo (p.ej., deletrear al revés)
- Lenguaje: denominación, repetición, comprensión
- Praxia constructiva: copia de una figura geométrica

| Puntuación | Interpretación Clínica |
|:---|:---|
| **27 – 30** | Rango normal. Sin deterioro apreciable. |
| **21 – 26** | Deterioro Cognitivo Leve (MCI) posible. |
| **11 – 20** | Demencia moderada. Dependencia funcional creciente. |
| **0 – 10** | Demencia grave. Alta dependencia en actividades de la vida diaria. |

⚠️ *El MMSE puede infraestimar el deterioro en pacientes con alta reserva cognitiva (educación elevada).*

---

### 2. CDR — Clinical Dementia Rating
El **CDR** es una herramienta semicuantitativa que evalúa la gravedad global de la demencia en 6 dominios:
memoria, orientación, juicio, vida social, hogar y cuidado personal.

| CDR | Estadio |
|:---|:---|
| **0** | Normal — Sin demencia. |
| **0.5** | Demencia muy leve / MCI — Dudoso. |
| **1** | Demencia leve. |
| **2** | Demencia moderada. |
| **3** | Demencia grave. |

*Un CDR de 0.5 no implica diagnóstico definitivo, pero es el umbral clínico para iniciar estudios adicionales.*

---

### 3. FAQ — Functional Activities Questionnaire
El **FAQ** cuantifica la capacidad del paciente para realizar **10 actividades instrumentales** de la vida diaria
(gestión de finanzas, cocinar, orientarse fuera de casa, etc.), puntuadas de 0 a 3 cada una.

| Puntuación Total | Interpretación |
|:---|:---|
| **0 – 5** | Funcionamiento independiente. |
| **6 – 9** | Deterioro funcional leve. |
| **≥ 9** | Dependencia funcional significativa. Criterio diagnóstico de demencia. |
        """)

    # ── SECCIÓN 3: Biomarcadores Estructurales MRI ─────────────────────────
    with st.expander("🖼️ Biomarcadores Estructurales (MRI — Volúmenes Normalizados)"):
        st.markdown("""
Los volúmenes cerebrales se extraen mediante segmentación automática con **FreeSurfer** y se normalizan
dividiéndolos por el **Volumen Intracraneal Total (TIV)** del paciente para eliminar el efecto del tamaño de la cabeza.
Se expresan como fracción del TIV (valores típicamente entre 0.001 y 0.100).

---

### 1. Hipocampo
El hipocampo es la estructura cerebral crítica para la **formación de nuevos recuerdos**. Es la primera región
afectada por la patología Tau en la EA, y su atrofia es el sello imaging de la enfermedad.

| Volumen Norm. | Interpretación |
|:---|:---|
| **> 0.005** | Normal. Sin atrofia significativa. |
| **0.003 – 0.005** | Atrofia leve-moderada. Sospecha de MCI o EA precoz. |
| **< 0.003** | Atrofia severa. Altamente sugestivo de EA establecida. |

---

### 2. Corteza Entorrinal
Es la **puerta de entrada** al hipocampo y una de las primeras regiones lesionadas por los ovillos neurofibrilares
(patología Tau). Su adelgazamiento precede años a los síntomas cognitivos.

| Volumen Norm. | Interpretación |
|:---|:---|
| **> 0.004** | Grosor cortical preservado. |
| **0.002 – 0.004** | Reducción moderada. Marcador de neurodegeneración temprana. |
| **< 0.002** | Pérdida severa. Correlaciona con estadios avanzados de Braak. |

---

### 3. Ventrículos Laterales
Los ventrículos son cavidades llenas de líquido cefalorraquídeo. Su **expansión** es una consecuencia directa de
la pérdida de tejido cerebral (neurodegeneración global), sirviendo como marcador indirecto de atrofia.

| Volumen Norm. | Interpretación |
|:---|:---|
| **< 0.025** | Normal para la edad. |
| **0.025 – 0.060** | Agrandamiento leve-moderado. Puede ser normal en mayores de 70 años. |
| **> 0.060** | Agrandamiento marcado. Indica pérdida global significativa de tejido. |
        """)

    # ── SECCIÓN 4: Sistema ATN (LCR) ────────────────────────────────────────
    with st.expander("🧪 Biomarcadores Moleculares — Sistema ATN (LCR)"):
        st.markdown("""
El **Sistema ATN**, propuesto por Jack et al. (NIA-AA, 2018), clasifica biológicamente la EA según tres
dimensiones de biomarcadores obtenibles mediante punción lumbar o PET:

| Dimensión | Biomarcador | Significado |
|:---|:---|:---|
| **A** — Amiloide | Aβ42 en LCR o PET-Amiloide | Presencia de placas de β-amiloide |
| **T** — Tau | pTau (Tau fosforilado) en LCR | Formación de ovillos neurofibrilares |
| **N** — Neurodegeneración | Tau Total o PET-FDG o MRI | Daño y muerte neuronal activa |

---

### 1. Aβ42 — Beta-Amiloide 42 (Marcador A)
La **formación de placas amiloides** en el parénquima cerebral secuestra el Aβ42 del LCR, por lo que
**valores BAJOS** en líquido cefalorraquídeo indican presencia de amiloide cerebral (**A+**).

| Aβ42 (pg/mL) | Estado ATN | Interpretación |
|:---|:---|:---|
| **> 1000** | A— | Sin evidencia de patología amiloide. |
| **900 – 1000** | A? | Zona gris. Requiere confirmación con PET. |
| **< 900** | **A+** | Depósito amiloide significativo. Alta especificidad para EA. |

---

### 2. Tau Total (Marcador N)
El **Tau total** en LCR es un marcador de **daño neuronal activo e inespecífico**. Aumenta en cualquier
condición que provoque muerte neuronal (EA, trauma, ACV).

| Tau Total (pg/mL) | Interpretación |
|:---|:---|
| **< 300** | Normal. Sin neurodegeneración activa significativa. |
| **300 – 450** | Elevación leve. Monitorizar. |
| **> 450** | **N+** — Daño neuronal activo. Criterio diagnóstico de EA. |

---

### 3. pTau — Tau Fosforilado (Marcador T)
El **pTau** (fosforilado en Treonina 181 o 217) es **específico de la EA**. Refleja la formación de
ovillos neurofibrilares intraneuronales, el segundo pilar patológico de la enfermedad.

| pTau (pg/mL) | Estado ATN | Interpretación |
|:---|:---|:---|
| **< 35** | T— | Sin patología Tau activa en EA. |
| **35 – 60** | T? | Zona de incertidumbre clínica. |
| **> 60** | **T+** | Formación de ovillos confirmada. Específico EA. |
        """)

    # ── SECCIÓN 5: Panel Demográfico y Genético ─────────────────────────────
    with st.expander("👤 Perfil Demográfico y Factor Genético (APOE4)"):
        st.markdown("""
### Edad
El mayor factor de riesgo no modificable para la EA. La prevalencia aumenta exponencialmente tras los 65 años.
El modelo tiene en cuenta la edad como variable continua para contextualizar los biomarcadores.

### Nivel Educativo
La **reserva cognitiva** acumulada durante la educación formal puede retrasar la manifestación clínica de la EA
aun cuando la patología biológica está presente. Un paciente con 20 años de educación y MMSE de 22 podría
tener mayor daño subyacente que uno con 8 años de educación y la misma puntuación.

### Género
Estadísticamente, las mujeres presentan mayor riesgo de EA, parcialmente relacionado con la longevidad diferencial
y factores hormonales posmenopáusicos.

### APOE4 — Apolipoproteína E Genotipo 4
El **APOE ε4** es el principal factor de riesgo genético conocido para la EA esporádica de inicio tardío.

| Genotipo | Riesgo Relativo EA |
|:---|:---|
| APOE2/2 (no portador) | Factor protector |
| APOE3/3 (más común) | Riesgo basal de referencia |
| APOE3/4 (portador 1 alelo) | ~3-4× mayor riesgo |
| APOE4/4 (portador 2 alelos) | ~8-12× mayor riesgo |

IMPORTANTE: Ser portador de APOE4 no implica desarrollar EA. Es un factor de riesgo probabilístico, no determinístico.
        """)

    # ── SECCIÓN 6: Motor de Diagnóstico ─────────────────────────────────────
    with st.expander("⚙️ Motor de Diagnóstico — Arquitectura XGBoost"):
        st.markdown("""
### Algoritmo: XGBoost (Extreme Gradient Boosting)
**XGBoost** es un algoritmo de ensamble basado en árboles de decisión que optimiza la función de pérdida
mediante **Gradient Boosting** secuencial. Cada nuevo árbol corrige los errores del conjunto anterior.

**Por qué XGBoost para datos médicos tabulares:**
- Domina en benchmarks con datos estructurados mixtos (numéricos + categóricos).
- Manejo nativo de valores faltantes (frecuentes en datos clínicos reales).
- Regularización L1/L2 incorporada que previene el sobreajuste.
- Alta interpretabilidad via SHAP Values, crítica en aplicaciones médicas.
- Demostrada superioridad en datasets ADNI y OASIS en literatura peer-reviewed.

### Dataset de Entrenamiento
- **ADNI (Alzheimer's Disease Neuroimaging Initiative):** Cohorte longitudinal multicéntrica de EEUU con más de
  2.000 participantes. Incluye MRI, PET, LCR, genómica y neuropsicología en múltiples visitas de seguimiento.
- **OASIS-3 (Open Access Series of Imaging Studies):** 1.378 participantes, >2.000 sesiones de neuroimagen.
  Open access (acceso libre) facilitado por Washington University in St. Louis.

### Features de Entrada (14 Variables)
`MMSE, CDR, FAQ, Género, Educación, Edad, APOE4, Hipocampo, Entorrinal, Temporal Medio, Ventrículos, Aβ42, Tau, pTau`

### Métricas de Rendimiento
| Métrica | Valor |
|:---|:---|
| **Accuracy** | ~88% |
| **F1-Score Macro** | ~0.87 |
| **AUC-ROC (clase AD)** | ~0.96 |
        """)

    # ── SECCIÓN 7: Escáner NLP ───────────────────────────────────────────────
    with st.expander("🔍 Escáner NLP — Extracción Automática de Informes Clínicos"):
        st.markdown("""
### ¿Cómo funciona el Escáner NLP?
El módulo de **Procesamiento de Lenguaje Natural (NLP)** implementa un motor de extracción de información
based en **expresiones regulares (Regex)** con tolerancia a variaciones lingüísticas clínicas en castellano.

**Proceso:**
1. El clínico pega el texto libre de un informe médico en el cuadro de texto.
2. El sistema busca patrones que correspondan a valores de biomarcadores (ej: `MMSE de 22`, `TAU: 480`, `Hipocampo 0.0038`).
3. Los valores extraídos se inyectan automáticamente en los sliders de la aplicación.
4. El médico puede corregir manualmente cualquier valor antes de ejecutar el modelo.

**Ejemplo de texto compatible:**
```
Paciente varón de 78 años, MMSE de 21 puntos, CDR de 1.
ABETA en LCR: 750 pg/mL. TAU: 520. PTAU de 72.
Hipocampo 0.0031, Entorrinal 0.0019, Ventrículos 0.065. APOE4 portador.
```

**Biomarcadores detectados automáticamente:** MMSE, CDR, FAQ, Edad, Aβ42, Tau, pTau, Hipocampo, Entorrinal, Ventrículos, APOE4.
        """)

    # ── SECCIÓN 8: Agente IA ──────────────────────────────────────────────────
    with st.expander("🤖 Agente de IA — Generación de Informes Clínicos"):
        st.markdown("""
### ¿Qué hace el Agente IA?
El agente transforma los datos numéricos en un **informe médico narrativo estructurado**, correlacionando
los biomarcadores siguiendo el sistema ATN de las guías NIA-AA 2018. Actúa como segundo lector especializado.

### Proveedores Disponibles

**🔵 OpenAI**
| Modelo | Velocidad | Calidad | Coste Aprox. |
|:---|:---|:---|:---|
| `gpt-4o-mini` | ⚡ Rápido | Alta | ~$0.0001/informe |
| `gpt-4o` | 🐢 Moderado | Muy Alta | ~$0.005/informe |
| `gpt-3.5-turbo` | ⚡ Rápido | Media | ~$0.00005/informe |

**🟢 Google Gemini**
| Modelo | Velocidad | Calidad | Coste Aprox. |
|:---|:---|:---|:---|
| `gemini-1.5-flash` | ⚡ Muy Rápido | Alta | Cuota gratuita disponible |
| `gemini-1.5-pro` | 🐢 Moderado | Muy Alta | ~$0.0035/1k tokens |
| `gemini-1.0-pro` | ⚡ Rápido | Media | Cuota gratuita disponible |

### ¿Cómo obtener una API Key?
- **OpenAI:** [platform.openai.com/api-keys](https://platform.openai.com/api-keys) — Requiere cuenta y método de pago.
- **Gemini:** [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) — **Capa gratuita disponible.**

### Seguridad
Las API Keys introducidas en esta interfaz **no se almacenan en el servidor** en ningún momento.
Permanecen únicamente en la memoria de la sesión de Streamlit, que se destruye al cerrar el navegador.
El usuario tiene control total sobre sus credenciales en todo momento.
        """)

    # ── SECCIÓN 9: Disclaimer ────────────────────────────────────────────────
    st.markdown("---")
    st.warning("""
    ⚕️ **Aviso de Uso Clínico Importante**

    NeuroNet-Fusion es una herramienta de **apoyo a la decisión clínica** diseñada para uso en investigación
    y formación académica. Los resultados producidos por este sistema **NO constituyen un diagnóstico médico
    definitivo** y no deben sustituir el juicio de un profesional sanitario cualificado.

    El diagnóstico definitivo de la Enfermedad de Alzheimer requiere evaluación clínica completa,
    confirmación mediante neuroimagen, análisis de LCR y seguimiento longitudinal por un especialista en neurología.
    """)
    st.info("📖 Proyecto Fin de Máster — IEBS Digital School | Inteligencia Artificial y Big Data | 2025")
