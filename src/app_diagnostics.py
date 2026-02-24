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

# CSS de Densidad Crítica
st.markdown("""
    <style>
    .block-container { padding-top: 0.5rem !important; padding-bottom: 0rem !important; max-width: 98% !important; }
    h1 { margin-top: -1rem !important; margin-bottom: 0.5rem !important; font-size: 2.2rem !important; }
    .stApp { background-color: #0E1117; color: white; }
    .medical-report { 
        background-color: #161B22; padding: 15px; border-radius: 8px; border: 1px solid #30363D;
        font-family: 'Consolas', monospace; color: #C9D1D9; font-size: 0.85rem;
    }
    h3 { font-size: 1.1rem !important; color: #58A6FF !important; margin-top: 10px !important; }
    .stSlider { padding-top: 5px !important; padding-bottom: 5px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURACIÓN DE AGENTE IA MULTIMODAL ---
st.sidebar.markdown("### 🤖 Configuración AI Agent")

# Selector de Proveedor
ai_provider = st.sidebar.selectbox("Proveedor de IA", ["OpenAI", "Google Gemini"], index=0)

if ai_provider == "OpenAI":
    env_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", value=env_api_key, type="password", placeholder="sk-...")
    selected_model = st.sidebar.selectbox("Modelo", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    st.sidebar.info("Modelos de OpenAI para razonamiento clínico avanzado.")
else:
    env_gemini_key = os.getenv("GEMINI_API_KEY", "")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", value=env_gemini_key, type="password", placeholder="AIza...")
    selected_model = st.sidebar.selectbox("Modelo", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"], index=0)
    st.sidebar.info("Google Gemini para ventanas de contexto amplias.")

# --- MOTOR DE EXTRACCIÓN NLP ---
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

# Inicializar estado si no existe
for k in ['mmse', 'age', 'abeta', 'tau', 'ptau', 'hippo', 'ento', 'vent', 'faq', 'apoe', 'cdr']:
    if k not in st.session_state:
        defaults = {'mmse': 24, 'age': 72, 'abeta': 850, 'tau': 450, 'ptau': 65, 'hippo': 0.0045, 'ento': 0.0028, 'vent': 0.08, 'faq': 10, 'apoe': 0, 'cdr': 0.5}
        st.session_state[k] = defaults.get(k)

# --- NAVEGACIÓN PRINCIPAL ---
tab_app, tab_doc = st.tabs(["🧠 NeuroNet-Fusion AI", "📚 Documentación Técnica"])

with tab_app:
    st.markdown("<h1 style='text-align: center;'>NeuroNet-Fusion</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0aec0; margin-top: -10px;'>Diagnóstico Multimodal de Precisión para Alzheimer</p>", unsafe_allow_html=True)

    if data_pkg:
        model, features = data_pkg['model'], data_pkg['features']
        
        # Scanner NLP
        with st.expander("🔍 Escáner Automático de Informes (NLP)", expanded=False):
            raw_text = st.text_area("Pega el informe clínico aquí:", placeholder="Ej: Varón de 75 años con MMSE de 22 puntos y amiloide de 800...")
            if st.button("🚀 Analizar Informe"):
                extracted = extract_from_text(raw_text)
                if extracted:
                    for k, v in extracted.items():
                        st.session_state[k] = v
                    st.success(f"Se han extraído {len(extracted)} biomarcadores correctamente.")
                else:
                    st.warning("No se detectaron biomarcadores. Revisa el formato.")

        # Layout Columnas
        col1, col2 = st.columns([1, 1], gap="medium")

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
                hippo = st.slider("Hipocampo (Vol. Norm.)", 0.001, 0.010, float(st.session_state.hippo), format="%.5f")
                ento = st.slider("Entorrinal (Vol. Norm.)", 0.001, 0.010, float(st.session_state.ento), format="%.5f")
                vent = st.slider("Ventrículos (Vol. Norm.)", 0.010, 0.100, float(st.session_state.vent), format="%.4f")
                
                c3, c4, c5 = st.columns(3)
                with c3: abeta = st.number_input("Aβ42", 200, 2000, int(st.session_state.abeta))
                with c4: tau = st.number_input("Tau T.", 50, 1500, int(st.session_state.tau))
                with c5: ptau_val = st.number_input("pTau", 10, 200, int(st.session_state.ptau))

        # --- PREDICCIÓN ---
        gender_val = 1 if gender == "Femenino" else 0
        # Mapeo exacto de 14 features según el modelo
        row_data = [bc_mmse, bc_cdr, bc_faq, gender_val, educat, age, apoe4, hippo, ento, 0.0185, vent, abeta, tau, ptau_val]
        
        input_data = pd.DataFrame([row_data], columns=features)
        probs = model.predict_proba(input_data)[0]
        class_idx = np.argmax(probs)
        lbls = ["Cognitivamente Normal (CN)", "Deterioro Cognitivo Leve (MCI)", "Enfermedad de Alzheimer (AD)"]
        colors = ["#10b981", "#f59e0b", "#ef4444"]

        st.markdown("---")
        st.markdown("### 📊 Resultado del Escáner")
        st.markdown(f"<h2 style='color: {colors[class_idx]}; text-align: center; border: 2px solid {colors[class_idx]}; padding: 10px; border-radius: 10px;'>{lbls[class_idx]}</h2>", unsafe_allow_html=True)
        
        # Gráfico Probs
        fig = go.Figure(go.Bar(
            x=probs*100, y=lbls, orientation='h',
            marker_color=colors, text=[f"{p*100:.1f}%" for p in probs], textposition='auto'
        ))
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

        # --- GENERACIÓN DE INFORME IA ---
        st.markdown("### 🤖 Generación de Informe Clínico Agéntico")
        if st.button(f"✨ Redactar Informe Final con {selected_model}", use_container_width=True):
            current_key = openai_api_key if ai_provider == "OpenAI" else gemini_api_key
            
            if not current_key:
                st.warning(f"Introduce tu API Key de {ai_provider} en la barra lateral.")
            else:
                try:
                    contexto_paciente = f"""
                    DATOS CLÍNICOS:
                    - Perfil: {age} años, {'Mujer' if gender_val==1 else 'Varón'}, {educat} años educ.
                    - APOE4: {'Portador' if apoe4==1 else 'No portador'}
                    - Cognición: MMSE {bc_mmse}/30, CDR {bc_cdr}, FAQ {bc_faq}/30
                    - MRI: Hipocampo {hippo:.5f}, Entorrinal {ento:.5f}, Ventrículos {vent:.4f}
                    - LCR: Aβ42 {abeta}, Tau {tau}, pTau {ptau_val}
                    - Predicción NeuroNet: {lbls[class_idx]} ({probs[class_idx]*100:.1f}%)
                    """
                    
                    system_prompt = "Eres un neurólogo experto. Genera un informe médico formal basado en los biomarcadores proporcionados, siguiendo el sistema ATN (NIA-AA). El tono debe ser profesional y detallado."
                    
                    with st.spinner("Analizando y redactando..."):
                        if ai_provider == "OpenAI":
                            client = OpenAI(api_key=openai_api_key)
                            response = client.chat.completions.create(
                                model=selected_model,
                                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": contexto_paciente}],
                                temperature=0.3
                            )
                            ai_report = response.choices[0].message.content
                        else:
                            genai.configure(api_key=gemini_api_key)
                            model_gem = genai.GenerativeModel(model_name=selected_model, system_instruction=system_prompt)
                            response = model_gem.generate_content(contexto_paciente)
                            ai_report = response.text
                        
                        st.markdown("#### 📄 Informe del Agente")
                        st.markdown(f"<div class='medical-report'>{ai_report}</div>", unsafe_allow_html=True)
                        st.download_button("📥 Descargar Informe (.md)", ai_report, file_name=f"Informe_NeuroNet_{age}.md", use_container_width=True)
                except Exception as e:
                    st.error(f"Error AI: {str(e)}")
    else:
        st.error("Error: No se pudo cargar el modelo de diagnóstico.")

with tab_doc:
    st.markdown("## 📚 Documentación Técnica")
    
    with st.expander("🛠️ Guía del Agente de IA", expanded=True):
        st.markdown("""
        NeuroNet-Fusion utiliza IA generativa para razonamiento clínico. 
        - **OpenAI:** GPT-4o recomendado para precisión diagnóstica.
        - **Gemini:** Ideal para análisis de largos historiales médicos.
        - **Seguridad:** Tus API Keys no se guardan en el servidor, solo en la memoria de la sesión actual.
        """)

    with st.expander("🧪 Glosario Médicos de Biomarcadores"):
        st.markdown("""
        | Biomarcador | Rango Normal | Significado Clínico |
        | :--- | :--- | :--- |
        | **MMSE** | 24 - 30 | Cognición Global preservada. |
        | **Aβ42** | > 900 pg/mL | Ausencia de placas de amiloide. |
        | **Hipocampo** | > 0.005 | Integridad estructural de la memoria. |
        | **CDR** | 0 | Ausencia de demencia clínica. |
        """)
        
    st.info("Esta herramienta es para uso académico y de apoyo a la investigación clínica.")
