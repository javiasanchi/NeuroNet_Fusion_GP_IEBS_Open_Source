import pandas as pd
import numpy as np
import os
import glob

print("🔄 Reconstruyendo Dataset Maestro con Macro-Análisis de MRI (Normalizado)...")

# --- Paths ---
BASE_PROJECT_DIR = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project'
ADNI_PATH = f'{BASE_PROJECT_DIR}/data/consolidated_analytical_v5.csv'
OASIS_CLINICAL_DIR = 'E:/MACHINE LEARNING/proyecto_global_IEBS/data/oasis3_clinical_tables'
OASIS_DATA_FILES = 'E:/MACHINE LEARNING/proyecto_global_IEBS/data/oasis3_extracted/OASIS3_data_files'
FS_FILE = 'E:/MACHINE LEARNING/proyecto_global_IEBS/data/oasis3_clinical_tables/OASIS3_data_files/SCANS/FS/csv/OASIS3_Freesurfer_output.csv'
OUTPUT_PATH = f'{BASE_PROJECT_DIR}/data/master_biomarker_v2_normalized.csv'

# --- 1. Procesar ADNI ---
adni = pd.read_csv(ADNI_PATH)
adni['target'] = adni['target'].replace({1: 0, 2: 1, 3: 2})

# Normalización por ICV en ADNI
# Nota: Asumimos que ADNI ya tiene estas columnas o sus equivalentes en v5
# Si faltan algunas, el modelo manejará los NaNs
adni['Hippo_Norm'] = adni['Hippocampus'] / adni['ICV']
adni['Ento_Norm'] = adni['Entorhinal'] / adni['ICV']
adni['MidTemp_Norm'] = adni['MidTemp'] / adni['ICV']
adni['Vent_Norm'] = adni['Ventricles'] / adni['ICV']

adni_cols = ['BCMMSE', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT', 'entry_age', 
             'APOE4_carrier', 'Hippo_Norm', 'Ento_Norm', 'MidTemp_Norm', 'Vent_Norm',
             'ABETA', 'TAU', 'PTAU', 'target']
adni_clean = adni[adni_cols].copy()
adni_clean['Centiloid'] = np.nan
adni_clean['source'] = 'ADNI'

# --- 2. Procesar OASIS-3 ---
print("⏳ Extrayendo volúmenes regionales de OASIS-3...")
df_fs = pd.read_csv(FS_FILE)

# Calcular promedios y normalizaciones
df_fs['Hippocampus_Total'] = df_fs['Left-Hippocampus_volume'] + df_fs['Right-Hippocampus_volume']
df_fs['Entorhinal_Total'] = df_fs['lh_entorhinal_volume'] + df_fs['rh_entorhinal_volume']
df_fs['MidTemp_Total'] = df_fs['lh_middletemporal_volume'] + df_fs['rh_middletemporal_volume']
df_fs['Ventricles_Total'] = df_fs['Left-Lateral-Ventricle_volume'] + df_fs['Right-Lateral-Ventricle_volume']

df_fs['Hippo_Norm'] = df_fs['Hippocampus_Total'] / df_fs['IntraCranialVol']
df_fs['Ento_Norm'] = df_fs['Entorhinal_Total'] / df_fs['IntraCranialVol']
df_fs['MidTemp_Norm'] = df_fs['MidTemp_Total'] / df_fs['IntraCranialVol']
df_fs['Vent_Norm'] = df_fs['Ventricles_Total'] / df_fs['IntraCranialVol']

# Agrupar por Sujeto (mediana)
oasis_mri = df_fs.groupby('Subject')[['Hippo_Norm', 'Ento_Norm', 'MidTemp_Norm', 'Vent_Norm']].median().reset_index()
oasis_mri = oasis_mri.rename(columns={'Subject': 'OASISID'})

# Datos Clínicos OASIS
demo_file = glob.glob(f'{OASIS_CLINICAL_DIR}/**/OASIS3_demographics.csv', recursive=True)[0]
df_demo = pd.read_csv(demo_file)
df_demo['APOE4_carrier'] = df_demo['APOE'].apply(lambda x: 1 if pd.notna(x) and '4' in str(int(x)) else 0)

cdr_file = glob.glob(f'{OASIS_CLINICAL_DIR}/**/OASIS3_UDSb4_cdr.csv', recursive=True)[0]
df_cdr = pd.read_csv(cdr_file)
df_cdr['target'] = df_cdr['CDRTOT'].map({0: 0, 0.5: 1, 1: 2, 2: 2, 3: 2}) # 1+ es AD

faq_file = glob.glob(f'{OASIS_CLINICAL_DIR}/**/OASIS3_UDSb7_faq_fas.csv', recursive=True)[0]
df_faq = pd.read_csv(faq_file)
faq_cols = ['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL']
for c in faq_cols: df_faq[c] = pd.to_numeric(df_faq[c], errors='coerce')
df_faq['BCFAQ'] = df_faq[faq_cols].sum(axis=1)

pet_path = f'{OASIS_DATA_FILES}/Centiloid/csv/OASIS3_amyloid_centiloid.csv'
df_pet = pd.read_csv(pet_path).rename(columns={'subject_id': 'OASISID', 'Centiloid_fSUVR_TOT_CORTMEAN': 'Centiloid'})
subj_pet = df_pet.groupby('OASISID')['Centiloid'].median().reset_index()

# Merge Final OASIS
o_merged = df_cdr.merge(df_demo[['OASISID', 'GENDER', 'EDUC', 'AgeatEntry', 'APOE4_carrier']], on='OASISID')
o_merged = o_merged.merge(df_faq[['OASIS_session_label', 'BCFAQ']], on='OASIS_session_label', how='left')
o_merged = o_merged.merge(oasis_mri, on='OASISID', how='left')
o_merged = o_merged.merge(subj_pet, on='OASISID', how='left')

o_clean = o_merged.rename(columns={'MMSE': 'BCMMSE', 'CDRSUM': 'BCCDR', 'GENDER': 'PTGENDER', 'EDUC': 'PTEDUCAT', 'AgeatEntry': 'entry_age'})
o_clean['ABETA'] = np.nan
o_clean['TAU'] = np.nan
o_clean['PTAU'] = np.nan
o_clean['source'] = 'OASIS-3'

# --- 3. Consolidar ---
master_v2 = pd.concat([adni_clean, o_clean[adni_cols + ['Centiloid', 'source']]], axis=0).reset_index(drop=True)
master_v2 = master_v2.dropna(subset=['target'])

# Limpieza final: Quitar inconsistencias (AD con MMSE 30 y CDR 0)
master_v2 = master_v2[~((master_v2['target'] == 2) & (master_v2['BCCDR'] == 0))]

master_v2.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Nuevo Dataset Bio-Normalizado guardado: {OUTPUT_PATH}")
print(f"📊 Total registros: {len(master_v2)}")
