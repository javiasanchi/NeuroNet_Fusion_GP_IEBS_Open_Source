import pandas as pd
import numpy as np
import os

ROOT = 'data/ida/ADNI'
BIO_ROOT = 'data/ida/ADNI/biospecimen_full'

def get_rid(sid):
    try:
        return int(sid.split('_')[-1])
    except:
        return None

def clean_numeric(x):
    if isinstance(x, str):
        # Handle cases like ">1700" or "<200"
        x = x.replace('>', '').replace('<', '')
    try:
        return float(x)
    except:
        return np.nan

print("🔄 Iniciando consolidación de datos con Biomarcadores SOTA...")

# 1. Load Core Data (Labels, Scores, Demog)
dx = pd.read_csv(os.path.join(ROOT, 'All_Subjects_DXSUM_16Feb2026.csv'))
scores = pd.read_csv(os.path.join(ROOT, 'All_Subjects_BLCHANGE_16Feb2026.csv'))
demog = pd.read_csv(os.path.join(ROOT, 'All_Subjects_PTDEMOG_16Feb2026.csv'))
entry = pd.read_csv(os.path.join(ROOT, 'All_Subjects_Study_Entry_16Feb2026.csv'))

# 2. Load New Biomarkers (CSF & Plasma)
upenn_csf = pd.read_csv(os.path.join(BIO_ROOT, 'UPENNBIOMK_MASTER_21Feb2026.csv'))
upenn_plasma = pd.read_csv(os.path.join(BIO_ROOT, 'UPENNPLASMA_21Feb2026.csv'))

# 3. Clean and Merge
# For biomarkers, we prefer Baseline ('bl' or 'm0')
# CSF
upenn_csf['ABETA'] = upenn_csf['ABETA'].apply(clean_numeric)
upenn_csf['TAU'] = upenn_csf['TAU'].apply(clean_numeric)
upenn_csf['PTAU'] = upenn_csf['PTAU'].apply(clean_numeric)
# Sort to get earliest visit per RID
upenn_csf = upenn_csf.sort_values(['RID', 'VISCODE']).drop_duplicates('RID')

# Plasma
upenn_plasma['AB42'] = upenn_plasma['AB42'].apply(clean_numeric)
upenn_plasma['AB40'] = upenn_plasma['AB40'].apply(clean_numeric)
upenn_plasma = upenn_plasma.sort_values(['RID', 'VISCODE']).drop_duplicates('RID')

# Core merges
dx = dx.sort_values('update_stamp').drop_duplicates('RID', keep='last')
scores = scores.sort_values('update_stamp').drop_duplicates('RID', keep='last')
demog = demog.sort_values('update_stamp').drop_duplicates('RID', keep='last')
entry['RID'] = entry['subject_id'].apply(get_rid)
entry = entry.dropna(subset=['RID']).drop_duplicates('RID')

# Big Merge
df = dx[['RID', 'DIAGNOSIS', 'DXNORM', 'DXMCI', 'DXAD']].merge(
    scores[['RID', 'BCMMSE', 'BCADAS', 'BCCDR', 'BCFAQ']], on='RID', how='inner'
).merge(
    demog[['RID', 'PTGENDER', 'PTEDUCAT']], on='RID', how='inner'
).merge(
    entry[['RID', 'entry_age']], on='RID', how='inner'
).merge(
    upenn_csf[['RID', 'ABETA', 'TAU', 'PTAU']], on='RID', how='left'
).merge(
    upenn_plasma[['RID', 'AB42', 'AB40']], on='RID', how='left'
)

# 4. Feature Engineering (As per the Google Doc)
# Abeta / Tau ratio
df['AB_TAU_RATIO'] = df['ABETA'] / (df['TAU'] + 1e-8)
df['PLASMA_RATIO'] = df['AB42'] / (df['AB40'] + 1e-8)

# Define Target
df['target'] = df['DIAGNOSIS'].fillna(df['DXNORM'].map({1:0})).fillna(df['DXMCI'].map({1:1})).fillna(df['DXAD'].map({1:2}))
df = df.dropna(subset=['target'])

# Save
output_path = 'Analytical_Biomarker_Project/data/consolidated_analytical_v2.csv'
df.to_csv(output_path, index=False)

print(f"✅ Datos consolidados con éxito en {output_path}")
print(f"📊 Total de muestras: {len(df)}")
print(f"🧪 Cobertura de CSF Abeta: {df['ABETA'].notna().sum()} / {len(df)}")
print(f"🧪 Cobertura de Plasma Ratio: {df['PLASMA_RATIO'].notna().sum()} / {len(df)}")
print("\nDistribución de clases:")
print(df['target'].value_counts())
