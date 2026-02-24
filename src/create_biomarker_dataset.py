import pandas as pd
import numpy as np
import os
import glob

def clean_numeric(x):
    try:
        return float(x)
    except:
        return np.nan

print("🔄 Creando Dataset Maestro con Biomarcadores (ADNI + OASIS-3)...")

# --- Paths ---
BASE_PROJECT_DIR = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project'
ADNI_PATH = f'{BASE_PROJECT_DIR}/data/consolidated_analytical_v5.csv'
OASIS_CLINICAL_DIR = 'E:/MACHINE LEARNING/proyecto_global_IEBS/data/oasis3_clinical_tables'
OASIS_DATA_FILES = 'E:/MACHINE LEARNING/proyecto_global_IEBS/data/oasis3_extracted/OASIS3_data_files'
OUTPUT_PATH = f'{BASE_PROJECT_DIR}/data/master_biomarker_dataset.csv'

# --- 1. Load ADNI (v5) ---
adni = pd.read_csv(ADNI_PATH)

# Harmonize ADNI targets: 1(CN), 2(MCI), 3(AD) -> 0, 1, 2
adni['target'] = adni['target'].replace({1: 0, 2: 1, 3: 2})

# Features for biomarker model
biomarker_cols = ['BCMMSE', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT', 'entry_age', 
                  'APOE4_carrier', 'Hippocampus', 'ABETA', 'TAU', 'PTAU', 'target']

adni_clean = adni[biomarker_cols].copy()
adni_clean['Centiloid'] = np.nan 
adni_clean['source'] = 'ADNI'

# --- 2. OASIS Processing ---
print("⏳ Procesando OASIS-3...")

# Demographics
demo_file = glob.glob(f'{OASIS_CLINICAL_DIR}/**/OASIS3_demographics.csv', recursive=True)[0]
df_demo = pd.read_csv(demo_file)
df_demo['PTGENDER'] = df_demo['GENDER']
df_demo['PTEDUCAT'] = df_demo['EDUC']
df_demo['entry_age'] = df_demo['AgeatEntry']
df_demo['APOE4_carrier'] = df_demo['APOE'].apply(lambda x: 1 if pd.notna(x) and '4' in str(int(x)) else 0)

# CDR
cdr_file = glob.glob(f'{OASIS_CLINICAL_DIR}/**/OASIS3_UDSb4_cdr.csv', recursive=True)[0]
df_cdr = pd.read_csv(cdr_file)
df_cdr = df_cdr.rename(columns={'MMSE': 'BCMMSE', 'CDRSUM': 'BCCDR'})
def map_oasis_target(cdrtot):
    if cdrtot == 0: return 0
    if cdrtot == 0.5: return 1
    if cdrtot >= 1.0: return 2
    return np.nan
df_cdr['target'] = df_cdr['CDRTOT'].apply(map_oasis_target)

# FAQ
faq_file = glob.glob(f'{OASIS_CLINICAL_DIR}/**/OASIS3_UDSb7_faq_fas.csv', recursive=True)[0]
df_faq = pd.read_csv(faq_file)
faq_cols = ['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL']
for c in faq_cols:
    df_faq[c] = pd.to_numeric(df_faq[c], errors='coerce')
df_faq['BCFAQ'] = df_faq[faq_cols].sum(axis=1)

# MRI
mri_file = glob.glob(f'{OASIS_CLINICAL_DIR}/**/OASIS3_Freesurfer_output.csv', recursive=True)[0]
df_vol = pd.read_csv(mri_file)
df_vol = df_vol.rename(columns={'TOTAL_HIPPOCAMPUS_VOLUME': 'Hippocampus', 'Subject': 'OASISID'})

# PET
pet_path = f'{OASIS_DATA_FILES}/Centiloid/csv/OASIS3_amyloid_centiloid.csv'
df_pet = pd.read_csv(pet_path)
df_pet = df_pet.rename(columns={'subject_id': 'OASISID', 'Centiloid_fSUVR_TOT_CORTMEAN': 'Centiloid'})

# Combine OASIS
oasis_combined = df_cdr.merge(df_demo[['OASISID', 'PTGENDER', 'PTEDUCAT', 'entry_age', 'APOE4_carrier']], on='OASISID', how='inner')
oasis_combined = oasis_combined.merge(df_faq[['OASIS_session_label', 'BCFAQ']], on='OASIS_session_label', how='left')

subj_vol = df_vol.groupby('OASISID')['Hippocampus'].median().reset_index()
subj_pet = df_pet.groupby('OASISID')['Centiloid'].median().reset_index()

o_merged = oasis_combined.merge(subj_vol, on='OASISID', how='left')
o_merged = o_merged.merge(subj_pet, on='OASISID', how='left')

o_merged['ABETA'] = np.nan
o_merged['TAU']   = np.nan
o_merged['PTAU']  = np.nan
o_merged['source'] = 'OASIS-3'

final_cols_with_source = biomarker_cols + ['Centiloid', 'source']
oasis_clean = o_merged[final_cols_with_source].copy()

# --- 3. Final Master ---
master_df = pd.concat([adni_clean, oasis_clean], axis=0).reset_index(drop=True)
master_df = master_df.dropna(subset=['target'])

for col in biomarker_cols + ['Centiloid']:
    if col != 'target':
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce')

master_df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Dataset Biomarcador guardado: {OUTPUT_PATH}")
print(f"📊 Estadísticas finales (N={len(master_df)}):")
print(f"   - ADNI: {len(adni_clean)}")
print(f"   - OASIS: {len(oasis_clean)}")
print("\nDisponibilidad de Biomarcadores:")
print(master_df.groupby('source')[['ABETA', 'TAU', 'PTAU', 'Centiloid']].count())
