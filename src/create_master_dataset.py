import pandas as pd
import numpy as np
import os
import glob

def clean_numeric(x):
    try:
        return float(x)
    except:
        return np.nan

print("🔄 Creando Dataset Maestro Armonizado (ADNI + OASIS-3)...")

# --- 1. Load ADNI (v5) ---
adni = pd.read_csv('Analytical_Biomarker_Project/data/consolidated_analytical_v5.csv')
# Harmonize ADNI targets: ADNI uses 1(CN), 2(MCI), 3(AD). Move to 0, 1, 2.
adni['target'] = adni['target'].replace({1: 0, 2: 1, 3: 2})
# Keep overlapping features only for maximum robustness
common_cols = ['BCMMSE', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT', 'entry_age', 'APOE4_carrier', 'Hippocampus', 'target']
adni_clean = adni[common_cols].copy()
adni_clean['source'] = 'ADNI'

# --- 2. Build OASIS Harmonized ---
CLINICAL_DIR = 'data/oasis3_clinical_tables'

# Demog
df_demo = pd.read_csv(glob.glob(f'{CLINICAL_DIR}/**/OASIS3_demographics.csv', recursive=True)[0])
df_demo['PTGENDER'] = df_demo['GENDER']
df_demo['PTEDUCAT'] = df_demo['EDUC']
df_demo['entry_age'] = df_demo['AgeatEntry']
df_demo['APOE4_carrier'] = df_demo['APOE'].apply(lambda x: 1 if pd.notna(x) and '4' in str(int(x)) else 0)

# CDR/MMSE
df_cdr = pd.read_csv(glob.glob(f'{CLINICAL_DIR}/**/OASIS3_UDSb4_cdr.csv', recursive=True)[0])
df_cdr = df_cdr.rename(columns={'MMSE': 'BCMMSE', 'CDRSUM': 'BCCDR'})
def map_oasis_target(cdrtot):
    if cdrtot == 0: return 0
    if cdrtot == 0.5: return 1
    if cdrtot >= 1.0: return 2
    return np.nan
df_cdr['target'] = df_cdr['CDRTOT'].apply(map_oasis_target)

# FAQ
df_faq = pd.read_csv(glob.glob(f'{CLINICAL_DIR}/**/OASIS3_UDSb7_faq_fas.csv', recursive=True)[0])
faq_cols = ['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL']
df_faq['BCFAQ'] = df_faq[faq_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)

# Volumes (Hippocampus)
df_vol = pd.read_csv(glob.glob(f'{CLINICAL_DIR}/**/OASIS3_Freesurfer_output.csv', recursive=True)[0])
df_vol = df_vol.rename(columns={'TOTAL_HIPPOCAMPUS_VOLUME': 'Hippocampus', 'Subject': 'OASISID'})

# Combine OASIS Parts
# Merge clinical parts
oasis_combined = df_cdr.merge(df_demo[['OASISID', 'PTGENDER', 'PTEDUCAT', 'entry_age', 'APOE4_carrier']], on='OASISID', how='inner')
oasis_combined = oasis_combined.merge(df_faq[['OASIS_session_label', 'BCFAQ']], on='OASIS_session_label', how='left')

# Convert relevant columns to numeric
cols_to_clean = ['BCMMSE', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT', 'entry_age', 'APOE4_carrier', 'Hippocampus']
for col in cols_to_clean:
    if col in oasis_combined.columns:
        oasis_combined[col] = pd.to_numeric(oasis_combined[col], errors='coerce')
    if col in df_vol.columns:
        df_vol[col] = pd.to_numeric(df_vol[col], errors='coerce')

# Volumes Grouping
subject_vols = df_vol.groupby('OASISID')['Hippocampus'].median().reset_index()
oasis_combined = oasis_combined.merge(subject_vols, on='OASISID', how='left')

oasis_clean = oasis_combined[common_cols].copy()
oasis_clean['source'] = 'OASIS-3'

# --- 3. Final Master Merge ---
master_df = pd.concat([adni_clean, oasis_clean], axis=0).reset_index(drop=True)
master_df = master_df.dropna(subset=['target'])

output_path = 'Analytical_Biomarker_Project/data/master_combined_dataset.csv'
master_df.to_csv(output_path, index=False)

print(f"✅ Dataset Maestro Armonizado guardado: {output_path}")
print(f"📊 Estadísticas:")
print(f"   - ADNI samples: {len(adni_clean)}")
print(f"   - OASIS samples: {len(oasis_clean)}")
print(f"   - Total: {len(master_df)}")
print("\nDistribución por Clase (Armonizado):")
print(master_df['target'].value_counts())
print("\nCobertura de Hippocampus:")
print(master_df.groupby('source')['Hippocampus'].count())
