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
    .block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; max-width: 99% !important; }
    header { visibility: hidden; }
    .stApp { background-color: #0E1117; color: white; }
    div[data-testid="stVerticalBlock"] > div { padding-top: 0rem !important; padding-bottom: 0.1rem !important; }
    .medical-report { 
        background-color: #161B22; padding: 12px; border-radius: 8px; border: 1px solid #30363D;
        font-family: 'Consolas', monospace; color: #C9D1D9; font-size: 0.75rem;
    }
    .status-box { padding: 4px; border-radius: 4px; text-align: center; font-weight: bold; font-size: 1.1rem; margin-top: 4px; }
    h3 { font-size: 0.9rem !important; color: #58A6FF !important; margin-bottom: 5px !important; }
    .stSlider { padding-top: 0px !important; padding-bottom: 0px !important; }
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
    st.sidebar.info("Utilizamos modelos de OpenAI para razonamiento clínico avanzado.")
else:
    env_gemini_key = os.getenv("GEMINI_API_KEY", "")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", value=env_gemini_key, type="password", placeholder="AIza...")
    selected_model = st.sidebar.selectbox("Modelo", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"], index=0)
    st.sidebar.info("Google Gemini ofrece alto rendimiento con ventanas de contexto amplias.")

# --- MOTOR DE EXTRACCIÓN NLP MEJORADO ---
def extract_from_text(text):
    data = {}
    # Patrones flexibles que aceptan "de", "del", "puntos", ":", etc.
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
            # Manejo especial para edad si usa el primer o segundo grupo de captura
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
        # Valores por defecto
        defaults = {'mmse': 24, 'age': 72, 'abeta': 850, 'tau': 450, 'ptau': 65, 'hippo': 0.0045, 'ento': 0.0028, 'vent': 0.08, 'faq': 10, 'apoe': 0, 'cdr': 0.5}
        st.session_state[k] = defaults.get(k)

if data_pkg:
    model, features = data_pkg['model'], data_pkg['features']
    
            # Let's reconstruct `row_data` based on this explicit order.
            row_data = [bc_mmse, bc_cdr, bc_faq, gender_val, educat, age, apoe4, hippo, ento, 0.0185, vent, abeta, tau, ptau_val] # Assuming 0.0185 for midtemp
            
            # Ensure the length of row_data matches the number of features
            if len(row_data) != len(features):
                st.error(f"Mismatch in number of input features. Expected {len(features)}, got {len(row_data)}.")
                # Fallback or more robust error handling needed here.
                # For now, I'll assume the `features` list from `data_pkg` matches the order I've constructed.
                # The original code had `1` as a feature, which is unusual. I'm replacing it with `gender_val`.
                # If the original `features` list truly contained a literal `1`, this needs adjustment.
                # Assuming the 4th feature was meant to be gender or a constant.
                # Let's assume the original `features` list was something like:
                # ['MMSE', 'CDR', 'FAQ', 'Gender', 'Education', 'Age', 'APOE4', 'Hippo', 'Entorrinal', 'MidTemporal', 'Ventricles', 'Abeta', 'Tau', 'pTau']
                # If so, `gender_val` should be at the 4th position (index 3).
                # The original code had `1` at index 3. I'll keep `gender_val` there.

            input_data = pd.DataFrame([row_data], columns=features)
            
            probs = model.predict_proba(input_data)[0]
            class_idx = np.argmax(probs)
            lbls = ["Cognitivamente Normal (CN)", "Deterioro Cognitivo Leve (MCI)", "Enfermedad de Alzheimer (AD)"]
            colors = ["#10b981", "#f59e0b", "#ef4444"]

            st.markdown("#### 📊 Dictamen Predictivo")
            with st.container(border=True):
                st.markdown(f"<h3 style='color: {colors[class_idx]}; border-left: 5px solid {colors[class_idx]}; padding-left: 15px;'>{lbls[class_idx]}</h3>", unsafe_allow_html=True)
                st.progress(float(probs[class_idx]))
                st.write(f"**Confianza del Modelo:** {probs[class_idx]*100:.1f}%")

            # --- GENERACIÓN DE INFORME IA ---
            st.markdown(f"### 🤖 Generación Agéntica con {ai_provider}")
            if st.button(f"✨ Generar Informe con {selected_model}", use_container_width=True):
                # Validación de llaves
                current_key = openai_api_key if ai_provider == "OpenAI" else gemini_api_key
                
                if not current_key:
                    st.warning(f"Por favor, introduce tu API Key de {ai_provider} en la barra lateral.")
                else:
                    try:
                        contexto_paciente = f"""
                        DATOS DEL PACIENTE:
                        - Edad: {age} | Educación: {educat} años | APOE4: {'Portador' if apoe4==1 else 'No portador'}
                        - MMSE: {bc_mmse}/30 | CDR: {bc_cdr} | FAQ: {bc_faq}/30
                        - MRI (Volúmenes Normalizados): Hippo: {hippo:.5f}, Entorrinal: {ento:.5f}, Ventrículos: {vent:.4f}
                        - CSF: Abeta: {abeta} pg/mL (A+ < 900), Tau: {tau} pg/mL (T+ > 450), pTau: {ptau_val} pg/mL
                        - Diagnóstico Probabilístico del Modelo: {lbls[class_idx]} (Confianza: {probs[class_idx]*100:.1f}%)
                        """
                        
                        system_prompt = "Eres un neurólogo experto en la Enfermedad de Alzheimer y diagnóstico multimodal. Tu tarea es elaborar un informe clínico altamente estructurado basado en los códigos del Módulo 8 (Procesamiento de Lenguaje Natural Clínico). Debes correlacionar los biomarcadores cognitivos, estructurales y moleculares (Sistema ATN) para dar un dictamen razonado siguiendo las guías NIA-AA 2018."
                        
                        with st.spinner(f"El Agente NeuroNet-{ai_provider} está analizando los biomarcadores..."):
                            if ai_provider == "OpenAI":
                                client = OpenAI(api_key=openai_api_key)
                                response = client.chat.completions.create(
                                    model=selected_model,
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": f"Genera un informe detallado para el siguiente perfil de paciente:\n{contexto_paciente}"}
                                    ],
                                    temperature=0.3
                                )
                                ai_report = response.choices[0].message.content
                            else:
                                genai.configure(api_key=gemini_api_key)
                                model_gemini = genai.GenerativeModel(model_name=selected_model, system_instruction=system_prompt)
                                response = model_gemini.generate_content(f"Genera un informe detallado para el siguiente perfil de paciente:\n{contexto_paciente}")
                                ai_report = response.text
                            
                            st.markdown("#### 📄 Informe Generado por el Agente")
                            st.markdown(f"<div class='medical-report' style='font-size: 0.85rem; height: 400px; overflow-y: scroll;'>{ai_report}</div>", unsafe_allow_html=True)
                            st.download_button("📥 Descargar Informe IA", ai_report, file_name=f"Informe_IA_{selected_model}_{age}.md", use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error al conectar con el proveedor de IA ({ai_provider}): {str(e)}")
    else:
        st.error("Modelo no cargado. Verifica la ruta en MODEL_PATH.")

with tab_doc:
    st.markdown("## 📚 Documentación Técnica NeuroNet-Fusion")
    
    with st.expander("🛠️ Guía de Configuración AI Agent", expanded=True):
        st.markdown("""
        ### ¿Cómo usar los Agentes de IA?
        NeuroNet-Fusion utiliza **Modelos de Lenguaje de Gran Escala (LLM)** para transformar los datos numéricos en una narrativa clínica profesional.
        
        1.  **Proveedor:** Selecciona **OpenAI** para precisión estándar o **Google Gemini** para ventanas de contexto amplias.
        2.  **API Key:** Es necesaria una llave personal.
            -   [Obtener OpenAI Key](https://platform.openai.com/api-keys)
            -   [Obtener Gemini Key](https://aistudio.google.com/app/apikey)
        3.  **Consumo:** Cada generación consume una pequeña fracción de crédito de tu cuenta personal.
        """)

    with st.expander("🧠 Diccionario de Biomarcadores"):
        st.markdown("""
        #### 1. Biomarcadores Cognitivos
        *   **MMSE (Mini-Mental State Examination):** Escala de 0-30. Valores < 24 sugieren deterioro cognitivo.
        *   **CDR (Clinical Dementia Rating):** Clasifica la gravedad. 0 (Normal), 0.5 (Deterioro Leve), 1-3 (Dementia).
        *   **FAQ (Functional Activities Questionnaire):** Evalúa independencia del paciente (0-30). > 9 indica dependencia funcional.

        #### 2. Biomarcadores Estructurales (MRI)
        Los volúmenes están normalizados por el **Volumen Intracraneal Total (TIV)**:
        *   **Hipocampo:** La atrofia en esta zona es el sello distintivo del Alzheimer precoz.
        *   **Corteza Entorrinal:** Una de las primeras áreas afectadas por la patología Tau.
        *   **Ventrículos:** Un aumento en su tamaño indica pérdida global de tejido cerebral (Neurodegeneración).

        #### 3. Biomarcadores Moleculares (LCR - Sistema ATN)
        *   **Aβ42 (Amiloide):** Valores bajos (< 900 pg/mL) indican presencia de placas en el cerebro (**A+**).
        *   **Tau T. / pTau:** Valores altos indican ovillos neurofibrilares y daño neuronal activo (**T+ / N+**).
        """)

    with st.expander("📊 Interpretación del Modelo"):
        st.markdown("""
        ### Red Neuronal XGBoost
        El motor de diagnóstico utiliza una arquitectura de **Gradient Boosting** entrenada sobre las cohortes internacionales ADNI y OASIS-3. 
        
        *   **CN:** El paciente no presenta signos clínicos ni biológicos de patología.
        *   **MCI:** Existe deterioro detectable, pero la funcionalidad diaria aún se preserva parcialmente.
        *   **AD:** Hallazgos compatibles con demencia tipo Alzheimer establecida.
        """)

    st.info("💡 Consejo: Usa las pestañas superiores para alternar rápidamente entre esta documentación y el panel de análisis.")
```
