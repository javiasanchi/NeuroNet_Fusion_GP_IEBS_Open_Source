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
from dotenv import load_dotenv

# Cargar variables de entorno (API Keys, Paths)
load_dotenv()

# Configuración ultra-ancha
st.set_page_config(page_title="NeuroNet-Fusion | AI Clinical Scanner", layout="wide", page_icon="🧠")

MODEL_PATH = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/models/neuro_fusion_final_v1.joblib'

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

# --- CONFIGURACIÓN DE AGENTE IA (OPENAI) ---
st.sidebar.markdown("### 🤖 Configuración AI Agent")
# Intentar obtener la clave de las variables de entorno como valor por defecto
env_api_key = os.getenv("OPENAI_API_KEY", "")
openai_api_key = st.sidebar.text_input("OpenAI API Key", value=env_api_key, type="password", placeholder="sk-...")
st.sidebar.info("Utilizamos GPT-4o-mini para lación agentica de informes clínicos siguiendo el Módulo 8 (NLP).")

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
    
    st.markdown("### 🧠 NeuroNet-Fusion | AI Clinical Intelligence")

    # --- ZONA DE ESCANEO AUTO-IA ---
    with st.expander("⚡ AUTO-SCAN: Pega aquí el Informe o Analítica para extracción automática", expanded=False):
        raw_text = st.text_area("Texto del informe clínico...", height=80, placeholder="Ej: Paciente con MMSE 22, Abeta de 450 y con atrofia en el Hipocampo de 0.0032...")
        if st.button("Procesar con IA"):
            extracted = extract_from_text(raw_text)
            for k, v in extracted.items():
                st.session_state[k] = v
            st.success(f"Se han actualizado {len(extracted)} parámetros desde el texto.")

    # --- FILA 1: ENTRADA DE DATOS (4 COLUMNAS) ---
    c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.2]) # Ajuste de anchura para MRI y CSF

    with c1:
        st.write("### 📋 Cognitivos")
        bc_mmse = st.number_input("MMSE", 0, 30, st.session_state.mmse, key="mm_in")
        bc_cdr = st.selectbox("CDR (Gravedad)", [0.0, 0.5, 1.0, 2.0], index=[0.0, 0.5, 1.0, 2.0].index(float(st.session_state.cdr)))
        bc_faq = st.slider("FAQ (Indep.)", 0, 30, int(st.session_state.faq))

    with c2:
        st.write("### 🧬 Perfil Paciente")
        age = st.number_input("Edad", 50, 100, int(st.session_state.age))
        apoe4 = st.radio("APOE4", [0, 1], index=int(st.session_state.apoe), format_func=lambda x: "Sí" if x==1 else "No", horizontal=True)
        educat = st.number_input("Educación (Años)", 0, 25, 12)

    with c3:
        st.write("### 🩻 MRI (v/ICV)")
        # Doble entrada para Hipocampo
        hippo_n = st.number_input("Hipocampo", 0.001, 0.010, float(st.session_state.hippo), format="%.5f", step=0.0001)
        hippo = st.slider("Slide: Hippo", 0.001, 0.010, hippo_n, format="%.4f", step=0.0001, label_visibility="collapsed")
        
        # Doble entrada para Entorrinal
        ento_n = st.number_input("Entorrinal", 0.001, 0.010, float(st.session_state.ento), format="%.5f", step=0.0001)
        ento = st.slider("Slide: Ento", 0.001, 0.010, ento_n, format="%.4f", step=0.0001, label_visibility="collapsed")
        
        # Temporal Medio
        midtemp = st.slider("Temporal Medio", 0.010, 0.050, 0.0185, format="%.3f", step=0.001)

    with c4:
        st.write("### 🔬 CSF & Ventr.")
        # Doble entrada para Ventrículos y Biomarcadores
        vent_n = st.number_input("Ventrículos", 0.01, 0.15, float(st.session_state.vent), format="%.4f", step=0.001)
        vent = st.slider("Slide: Vent", 0.01, 0.15, vent_n, format="%.3f", step=0.001, label_visibility="collapsed")
        
        abeta = st.number_input("ABETA pg/mL", 0, 2000, int(st.session_state.abeta))
        tau = st.number_input("TAU pg/mL", 0, 1000, int(st.session_state.tau))

    # Inferencia - ptau se mantiene constante o se puede añadir si es necesario
    input_data = pd.DataFrame([[bc_mmse, bc_cdr, bc_faq, 1, educat, age, apoe4, hippo, ento, midtemp, vent, abeta, tau, st.session_state.ptau]], columns=features)
    probs = model.predict_proba(input_data)[0]
    class_idx = np.argmax(probs)
    lbls, clrs = ["SANO (CN)", "DEBIL (MCI)", "ALZHEIMER (AD)"], ["#28a745", "#ffc107", "#dc3545"]

    # --- FILA 2: ANALÍTICA ---
    st.divider()
    res1, res2, res3 = st.columns([1, 1.2, 1])
    with res1:
        st.markdown(f"<div class='status-box' style='background-color: {clrs[class_idx]}; color: black;'>{lbls[class_idx]}</div>", unsafe_allow_html=True)
        st.metric("Confianza IA", f"{probs[class_idx]*100:.1f}%")
    with res2:
        fig_prob = px.bar(x=lbls, y=probs, color=lbls, color_discrete_sequence=clrs, height=130)
        fig_prob.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig_prob, use_container_width=True, config={'displayModeBar': False})
    with res3:
        fig_radar = go.Figure(data=go.Scatterpolar(r=[hippo*100, ento*100, midtemp*25, vent*8], theta=['Hippo', 'Ento', 'TempM', 'Vent'], fill='toself'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), height=160, margin=dict(l=30, r=30, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

    # --- FILA 3: INFORME NEUROLÓGICO NARRATIVO ---
    with st.expander("🔬 VER DICTAMEN MÉDICO INTEGRAL Y ANÁLISIS MULTIMODAL", expanded=False):

        # ── Lógica de interpretación clínica ──────────────────────────────
        ami_stat     = "A+" if abeta < 900 else "A-"
        tau_stat     = "T+" if tau > 450 else "T-"
        atrophy_stat = "N+" if hippo < 0.0048 else "N-"
        ptau_val     = st.session_state.ptau

        # Clasificación ATN cruzada con diagnóstico IA
        if class_idx == 2:
            ea_tipo = "Enfermedad de Alzheimer Probable — Estadio Demencia"
            ea_fase = ("demencia establecida de grado leve-moderado" if bc_cdr <= 1
                       else "demencia moderada-severa con afectación funcional significativa")
        elif class_idx == 1:
            ea_tipo = "Deterioro Cognitivo Leve (MCI) — Estadio Prodromal"
            ea_fase = "fase prodrómica o de conversión, con riesgo elevado de progresión a demencia franca"
        else:
            ea_tipo = "Sin Criterios de Enfermedad de Alzheimer — Cognitivamente Normal (CN)"
            ea_fase = "estado cognitivo dentro de la normalidad para la edad y nivel educativo"

        # Narrativas por sección
        mmse_text = (
            f"El paciente obtiene una puntuación de <b>{bc_mmse}/30</b> en el Minimental State Examination (MMSE). "
            f"{'Una puntuación inferior a 24 puntos se considera indicativa de deterioro cognitivo clínico, con afectación objetivable en orientación, memoria y funciones ejecutivas. En este caso, el score apunta a un ' + ('deterioro leve-moderado.' if bc_mmse >= 18 else 'deterioro moderado-severo con importante pérdida de autonomía.') if bc_mmse < 24 else 'Este valor se sitúa dentro del umbral de normalidad, aunque en presencia de otros marcadores positivos debe interpretarse con cautela, pudiendo enmascarar un deterioro incipiente no captado por el MMSE aislado.'}"
        )
        cdr_text = (
            f"La escala CDR (<i>Clinical Dementia Rating</i>) puntúa <b>{bc_cdr}</b>. "
            + {0: "Una CDR de 0 implica que no existe demencia clínicamente detectable.",
               0.5: "Una CDR de 0.5 refleja <b>demencia cuestionable o MCI</b>: el paciente muestra déficit de memoria leve pero conserva independencia funcional casi completa. Es el estadio típico de la fase prodrómica.",
               1.0: "Una CDR de 1 corresponde a <b>demencia leve</b>: pérdida de memoria moderada, desorientación temporal y dificultades en resolución de problemas complejos, aunque con cierta autonomía preservada.",
               2.0: "Una CDR de 2 indica <b>demencia moderada</b>: el paciente requiere supervisión continua para las actividades de la vida diaria, presenta desorientación espacio-temporal y notables deficiencias lingüísticas."
               }.get(float(bc_cdr), "Valor CDR fuera de rango estándar.")
        )
        faq_text = (
            f"La escala FAQ (<i>Functional Activities Questionnaire</i>) puntúa <b>{bc_faq}/30</b>. "
            f"{'Una puntuación superior a 9 es diagnóstica de deterioro funcional relevante. Con ' + str(bc_faq) + ' puntos, el paciente presenta ' + ('un compromiso funcional severo que le impide gestionar tareas básicas como la economía doméstica, medicación o comunicación autónoma.' if bc_faq > 15 else 'un compromiso funcional moderado: puede realizar actividades básicas con supervisión, pero ha perdido autonomía en tareas instrumentales complejas.') if bc_faq > 9 else 'Con ' + str(bc_faq) + ' puntos, el impacto funcional es leve o nulo, lo que contradice parcialmente un diagnóstico de demencia establecida si los demás marcadores no son concluyentes.'}"
        )
        hippo_text = (
            f"El volumen hipocampal normalizado por volumen intracraneal estimado (ICV) es de <b>{hippo:.5f}</b>. "
            f"El hipocampo es la estructura neuronal central en la formación y consolidación de nuevos recuerdos. Su atrofia constituye uno de los marcadores estructurales más precoces y específicos de la Enfermedad de Alzheimer. "
            f"{'El índice actual refleja una <b>atrofia hipocampal SEVERA</b> (valores normales ≥ 0.0048), lo que indica una pérdida neuronal masiva en la región CA1 y el subículo, correlacionada directamente con la incapacidad para la memoria episódica de nueva adquisición.' if hippo < 0.0035 else '<b>Atrofia hipocampal MODERADA</b>: pérdida significativa de volumen respecto a controles sanos de la misma edad, compatible con un estadio de conversión MCI-a-Demencia.' if hippo < 0.0048 else 'El volumen hipocampal se sitúa dentro de rangos relativamente preservados, aunque debe correlacionarse con la clínica y los biomarcadores moleculares.'}"
        )
        ento_text = (
            f"La corteza entorrinal normalizada registra un índice de <b>{ento:.5f}</b>. "
            f"Esta región actúa como puerta de entrada al hipocampo y es, frecuentemente, la primera zona afectada en la cascada amiloide. Su adelgazamiento precede incluso a la aparición de síntomas clínicos y es un indicador de patología en estadio Braak I-II. "
            f"{'El valor actual indica un <b>adelgazamiento entorrinal marcado</b>, congruente con la progresión de la patología tau desde la región transentorrinal hacia el hipocampo.' if ento < 0.004 else 'El volumen entorrinal muestra una reducción leve-moderada, compatible con los estadios iniciales de la enfermedad.'}"
        )
        vent_text = (
            f"El índice de dilatación ventricular normalizado es <b>{vent:.4f}</b>. "
            f"La ampliación del sistema ventricular (<i>hidrocefalia ex-vacuo</i>) es un fenómeno secundario a la pérdida progresiva de parénquima cortical y subcortical. No es patognomónica de EA, pero su presencia correlaciona con la magnitud de la atrofia global. "
            f"{'El valor actual evidencia una <b>dilatación ventricular significativa</b>, coherente con la pérdida de volumen cerebral observada en las regiones temporales.' if vent > 0.09 else 'La dilatación ventricular es leve-moderada, indicando una atrofia difusa presente pero no extrema.'}"
        )
        abeta_text = (
            f"La concentración de <b>β-Amiloide 42 en LCR es de {abeta} pg/mL</b>. "
            f"En condiciones normales, el Aβ42 circula libremente en el líquido cefalorraquídeo. Cuando existe patología amiloide, las proteínas se agregan en placas intraparenquimatosas, reduciendo los niveles detectables en LCR. El umbral clínico de positividad se establece en <b>&lt;900 pg/mL</b>. "
            f"{'Con <b>' + str(abeta) + ' pg/mL el perfil amiloide es POSITIVO (A+)</b>: existe depósito significativo de placas de amiloide en el parénquima cerebral. Esto coloca al paciente en la fase biológica de la EA según el marco NIA-AA 2018, independientemente de la sintomatología.' if abeta < 900 else 'Con <b>' + str(abeta) + ' pg/mL el perfil amiloide es NEGATIVO (A-)</b>: los niveles de Aβ42 en LCR son suficientes, lo que reduce la probabilidad de patología amiloide activa significativa.'}"
        )
        tau_text = (
            f"La <b>proteína Tau total asciende a {tau} pg/mL</b> y la <b>Tau hiperfosforilada (pTau) a {ptau_val} pg/mL</b>. "
            f"La proteína Tau es el componente estructural principal de los ovillos neurofibrilares (ONF), lesión histológica canónica del Alzheimer. Su hiperfosforilación desestabiliza los microtúbulos axonales, provocando muerte neuronal y propagación transináptica de la patología. Los valores de referencia se sitúan en <b>Tau total &lt;450 pg/mL y pTau &lt;61 pg/mL</b>. "
            f"{'Ambas fracciones están <b>ELEVADAS (T+)</b>: se confirma tauopatía activa con neurodegeneración en curso. La relación pTau/Tau sugiere un patrón de hiperfosforilación compatible con estadios Braak III-V.' if tau > 450 else 'Los valores de Tau se sitúan <b>dentro o próximos a la normalidad (T-)</b>, lo que atenúa la confirmación biológica de tauopatía activa, aunque no la descarta en fases muy precoces.'}"
        )
        apoe_text = (
            f"El perfil genético del paciente {'es <b>PORTADOR del alelo APOE ε4</b>' if apoe4==1 else 'es <b>NO PORTADOR del alelo APOE ε4</b>'}. "
            f"{'El alelo APOE4 constituye el principal factor de riesgo genético de la Enfermedad de Alzheimer esporádica: multiplica el riesgo por 3-4 en heterocigotos y por 8-12 en homocigotos, y adelanta la edad de inicio entre 5-10 años. Además, se asocia a una mayor carga de placas amiloides y un aclaramiento reducido del Aβ42.' if apoe4==1 else 'La ausencia del alelo APOE4 no excluye el diagnóstico de EA, pero indica que la patología, de confirmarse, tiene una base genética menos predisponente y es más probable que su inicio sea de inicio tardío sin factor hereditario de alto riesgo.'}"
        )

        # Conclusión sintetizadora
        conclusion_text = (
            f"Tras el análisis integrado de los 14 biomarcadores clínicos, estructurales y moleculares proporcionados, el sistema NeuroNet-Fusion —basado en un modelo XGBoost entrenado sobre la cohorte ADNI+OASIS-3 con normativa ATN— clasifica a este paciente como: "
            f"<b style='color:{clrs[class_idx]};font-size:1.1rem;'>{ea_tipo}</b>.<br><br>"
            f"Clínicamente, el perfil es compatible con una <b>{ea_fase}</b>. "
            f"La convergencia de los marcadores {ami_stat}/{tau_stat}/{atrophy_stat} en el marco ATN {'confirma biológicamente la presencia de la cascada fisiopatológica de la EA, con carga amiloide establecida, neurodegeneración activa mediada por Tau y atrofia estructural objetivable. Se recomienda <b>inicio de protocolo de seguimiento neurológico intensivo, valoración de elegibilidad para ensayos de inmunoterapia anti-amiloide y comunicación diagnóstica con el paciente y familia</b> conforme a los protocolos de la Sociedad Española de Neurología (SEN).' if ami_stat=='A+' and tau_stat=='T+' else 'sugiere una fase biológica intermedia con componente neurodegenerativo presente pero sin confirmación amiloide completa. Se recomienda <b>repetición de los biomarcadores en 12-18 meses y seguimiento neuropsicológico</b>.' if class_idx==1 else 'se mantiene dentro de la normalidad biológica. No obstante, la edad del paciente y el perfil genético aconsejan <b>seguimiento preventivo anual con escalas cognitivas</b>.'}"
        )

        # ── Render HTML inline ─────────────────────────────────────────────
        ami_bg = "#442a2a" if ami_stat == "A+" else "#213028"
        ami_cl = "#ff7b72" if ami_stat == "A+" else "#7ee787"
        tau_bg = "#442a2a" if tau_stat == "T+" else "#213028"
        tau_cl = "#ff7b72" if tau_stat == "T+" else "#7ee787"
        atr_bg = "#442a2a" if atrophy_stat == "N+" else "#213028"
        atr_cl = "#ff7b72" if atrophy_stat == "N+" else "#7ee787"

        S = "font-family:'Segoe UI',sans-serif;color:#C9D1D9;font-size:0.88rem;line-height:1.7;"
        H = "color:#58A6FF;font-size:1rem;font-weight:bold;margin:18px 0 6px 0;border-left:4px solid #58A6FF;padding-left:10px;"
        V = "font-weight:bold;color:#E6EDF3;background:#21262D;padding:2px 8px;border-radius:4px;font-size:0.85rem;"

        st.markdown(f"""
        <div style="background:linear-gradient(160deg,#161B22,#0D1117);border:1px solid #30363D;border-radius:14px;padding:28px;{S}">

          <!-- Cabecera -->
          <div style="border-bottom:2px solid #58A6FF;padding-bottom:14px;margin-bottom:24px;display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <div style="font-size:1.3rem;font-weight:bold;color:white;letter-spacing:0.5px;">🧠 NEURONET-FUSION v1.0</div>
              <div style="color:#8B949E;font-size:0.78rem;margin-top:4px;">
                INFORME NEUROLÓGICO CLÍNICO ASISTIDO POR IA &nbsp;|&nbsp; ID: NF-{age}{datetime.now().second} &nbsp;|&nbsp; {datetime.now().strftime('%d/%m/%Y %H:%M')} &nbsp;|&nbsp; Edad: {age} años &nbsp;|&nbsp; Escolaridad: {educat} años
              </div>
            </div>
            <div style="text-align:right;">
              <span style="padding:5px 14px;border-radius:20px;font-weight:bold;font-size:0.82rem;background:{ami_bg};color:{ami_cl};border:1px solid {ami_cl};">{ami_stat} Amiloide</span>&nbsp;
              <span style="padding:5px 14px;border-radius:20px;font-weight:bold;font-size:0.82rem;background:{tau_bg};color:{tau_cl};border:1px solid {tau_cl};">{tau_stat} Tau</span>&nbsp;
              <span style="padding:5px 14px;border-radius:20px;font-weight:bold;font-size:0.82rem;background:{atr_bg};color:{atr_cl};border:1px solid {atr_cl};">{atrophy_stat} Neurodeg.</span>
            </div>
          </div>

          <!-- I. COGNITIVO -->
          <div style="{H}">I. EVALUACIÓN NEUROPSICOLÓGICA Y FUNCIONAL</div>

          <div style="margin-bottom:10px;"><span style="{V}">MMSE: {bc_mmse}/30</span>&nbsp; {mmse_text}</div>
          <div style="margin-bottom:10px;"><span style="{V}">CDR: {bc_cdr}</span>&nbsp; {cdr_text}</div>
          <div style="margin-bottom:10px;"><span style="{V}">FAQ: {bc_faq}/30</span>&nbsp; {faq_text}</div>

          <!-- II. MRI -->
          <div style="{H}">II. NEUROIMAGEN ESTRUCTURAL — VOLUMETRÍA MRI (Normalizada por ICV)</div>

          <div style="margin-bottom:10px;"><span style="{V}">Hipocampo/ICV: {hippo:.5f}</span>&nbsp; {hippo_text}</div>
          <div style="margin-bottom:10px;"><span style="{V}">Entorrinal/ICV: {ento:.5f}</span>&nbsp; {ento_text}</div>
          <div style="margin-bottom:10px;"><span style="{V}">Ventrículos/ICV: {vent:.4f}</span>&nbsp; {vent_text}</div>

          <!-- III. MOLECULAR -->
          <div style="{H}">III. BIOMARCADORES MOLECULARES EN LCR — PERFIL ATN</div>

          <div style="margin-bottom:10px;"><span style="{V}">Aβ42: {abeta} pg/mL &nbsp;|&nbsp; pTau: {ptau_val} pg/mL &nbsp;|&nbsp; Tau Total: {tau} pg/mL</span></div>
          <div style="margin-bottom:10px;">{abeta_text}</div>
          <div style="margin-bottom:10px;">{tau_text}</div>
          <div style="margin-bottom:10px;"><span style="{V}">APOE4: {'PORTADOR' if apoe4==1 else 'NO PORTADOR'}</span>&nbsp; {apoe_text}</div>

          <!-- IV. CONCLUSIÓN -->
          <div style="{H}">IV. SÍNTESIS DIAGNÓSTICA Y RECOMENDACIÓN CLÍNICA</div>
          <div style="background:#0d1117;border-left:5px solid {clrs[class_idx]};padding:18px 20px;border-radius:8px;margin-top:8px;">
            {conclusion_text}
            <div style="margin-top:14px;font-size:0.78rem;color:#8B949E;border-top:1px solid #30363D;padding-top:10px;">
              Documento generado automáticamente. No sustituye al criterio clínico del especialista. &nbsp;·&nbsp;
              NeuroNet-Fusion &nbsp;·&nbsp; IEBS Business School — Proyecto Final Postgrado IA &nbsp;·&nbsp; {datetime.now().strftime('%Y')}
            </div>
          </div>

        </div>
        """, unsafe_allow_html=True)

        # Informe texto para descarga
        informe_txt = (
            f"NEURONET-FUSION | DICTAMEN NEUROLÓGICO CLÍNICO\n"
            f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')} | ID: NF-{age}{datetime.now().second}\n"
            f"Edad: {age} años | Escolaridad: {educat} años | APOE4: {'Portador' if apoe4==1 else 'No portador'}\n"
            f"{'='*70}\n\n"
            f"DIAGNÓSTICO IA : {ea_tipo}\n"
            f"CONFIANZA      : {probs[class_idx]*100:.2f}%\n"
            f"PERFIL ATN     : {ami_stat} / {tau_stat} / {atrophy_stat}\n\n"
            f"I. NEUROPSICOLÓGICO\n"
            f"   MMSE {bc_mmse}/30 | CDR {bc_cdr} | FAQ {bc_faq}/30\n\n"
            f"II. MRI VOLUMÉTRÍA\n"
            f"   Hipocampo/ICV: {hippo:.5f} | Entorrinal/ICV: {ento:.5f} | Ventrículos/ICV: {vent:.4f}\n\n"
            f"III. BIOMARCADORES LCR\n"
            f"   Aβ42: {abeta} pg/mL | Tau Total: {tau} pg/mL | pTau: {ptau_val} pg/mL\n\n"
            f"{'='*70}\n"
            f"IEBS Business School — Proyecto Final Postgrado Inteligencia Artificial\n"
        )
        st.download_button("💾 Descargar Dictamen Clínico Completo", informe_txt,
                           file_name=f"Dictamen_NF_{age}.txt", use_container_width=True)

        # --- SECCIÓN AGENTICA (NLP MODULO 8) ---
        st.divider()
        st.markdown("### 🤖 Generación Agéntica de Informe (IA Avanzada)")
        if st.button("✨ Generar Informe NeuroNet-IA (GPT-4o-mini)", use_container_width=True):
            if not openai_api_key:
                st.warning("Por favor, introduce tu OpenAI API Key en la barra lateral.")
            else:
                try:
                    client = OpenAI(api_key=openai_api_key)
                    
                    # Consolidación de datos para el prompt
                    contexto_paciente = f"""
                    DATOS DEL PACIENTE:
                    - Edad: {age} | Educación: {educat} años | APOE4: {'Portador' if apoe4==1 else 'No portador'}
                    - MMSE: {bc_mmse}/30 | CDR: {bc_cdr} | FAQ: {bc_faq}/30
                    - MRI (Volúmenes Normalizados): Hippo: {hippo:.5f}, Entorrinal: {ento:.5f}, Ventrículos: {vent:.4f}
                    - CSF: Abeta: {abeta} pg/mL (A+ < 900), Tau: {tau} pg/mL (T+ > 450), pTau: {ptau_val} pg/mL
                    - Diagnóstico Probabilístico del Modelo: {lbls[class_idx]} (Confianza: {probs[class_idx]*100:.1f}%)
                    """
                    
                    with st.spinner("El Agente NeuroNet-IA está analizando los biomarcadores..."):
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "Eres un neurólogo experto en la Enfermedad de Alzheimer y diagnóstico multimodal. Tu tarea es elaborar un informe clínico altamente estructurado basado en los códigos del Módulo 8 (Procesamiento de Lenguaje Natural Clínico). Debes correlacionar los biomarcadores cognitivos, estructurales y moleculares (Sistema ATN) para dar un dictamen razonado siguiendo las guías NIA-AA 2018."},
                                {"role": "user", "content": f"Genera un informe detallado para el siguiente perfil de paciente:\n{contexto_paciente}"}
                            ],
                            temperature=0.3
                        )
                        ai_report = response.choices[0].message.content
                        
                        st.markdown("#### 📄 Informe Generado por el Agente")
                        st.markdown(f"<div class='medical-report' style='font-size: 0.85rem; height: 400px; overflow-y: scroll;'>{ai_report}</div>", unsafe_allow_html=True)
                        st.download_button("📥 Descargar Informe IA", ai_report, file_name=f"Mantenimiento_IA_NF_{age}.md", use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error al conectar con OpenAI: {str(e)}")

else:
    st.error("Modelo no cargado.")


