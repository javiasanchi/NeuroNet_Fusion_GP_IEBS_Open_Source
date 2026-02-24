import pandas as pd
import numpy as np
import os
import glob

def clean_numeric(x):
    try:
        return float(x)
    except:
        return np.nan

print("🔄 Consolidando datos de OASIS-3 para validación técnica...")

# Paths
CLINICAL_DIR = 'data/oasis3_clinical_tables'

# Load Demographics
demo_file = glob.glob(f'{CLINICAL_DIR}/**/OASIS3_demographics.csv', recursive=True)[0]
df_demo = pd.read_csv(demo_file)

# Load CDR/MMSE
cdr_file = glob.glob(f'{CLINICAL_DIR}/**/OASIS3_UDSb4_cdr.csv', recursive=True)[0]
df_cdr = pd.read_csv(cdr_file)

# --- Processing Demographics ---
# Mapping columns: 
# ADNI [PTGENDER, PTEDUCAT, entry_age, APOE4_carrier]
# OASIS [GENDER, EDUC, AgeatEntry, APOE]

df_demo['PTGENDER'] = df_demo['GENDER']
df_demo['PTEDUCAT'] = df_demo['EDUC']
df_demo['entry_age'] = df_demo['AgeatEntry']
df_demo['APOE4_carrier'] = df_demo['APOE'].apply(lambda x: 1 if pd.notna(x) and '4' in str(int(x)) else 0)

# --- Processing CDR/MMSE ---
# Mapping columns:
# ADNI [BCMMSE, BCCDR] -> MMSE, CDRSUM
# target -> based on CDRTOT

df_cdr = df_cdr.rename(columns={'MMSE': 'BCMMSE', 'CDRSUM': 'BCCDR'})

# Define Target (OASIS mapping)
def map_oasis_target(cdrtot):
    if cdrtot == 0: return 0 # CN
    if cdrtot == 0.5: return 1 # MCI
    if cdrtot >= 1.0: return 2 # AD
    return np.nan

df_cdr['target'] = df_cdr['CDRTOT'].apply(map_oasis_target)

# Sort by visit and take latest or earliest per subject?
# For validation, we can take all visits or just the first. Let's take all valid clinical visits.
df_cdr = df_cdr.dropna(subset=['target'])

# --- Merge ---
df = df_cdr.merge(df_demo[['OASISID', 'PTGENDER', 'PTEDUCAT', 'entry_age', 'APOE4_carrier']], on='OASISID', how='inner')

# Add placeholders for missing features in OASIS (ADAS, FAQ, Volumes)
# This is necessary because the ADNI-trained model expects these features.
# We will use the median from ADNI or 0/NaN depending on the model's tolerance.
# XGBoost handles NaNs well, but let's be explicit if we want to fill them.
# For now, let's keep them as NaN and let XGBoost handle them.

missing_features = [
    'BCADAS', 'BCFAQ', 'Hippocampus', 'Entorhinal', 'ABETA', 'TAU', 'PTAU', 
    'CSF_AB_TAU_RATIO', 'TRAASCOR', 'TRABSCOR', 'BNTTOTAL', 'CATANIMSC', 'PHS'
]
for feat in missing_features:
    df[feat] = np.nan

# Save
output_path = 'Analytical_Biomarker_Project/data/consolidated_oasis3.csv'
df.to_csv(output_path, index=False)

print(f"✅ Dataset OASIS-3 consolidado en {output_path}")
print(f"📊 Total de muestras: {len(df)}")
print(df['target'].value_counts())
