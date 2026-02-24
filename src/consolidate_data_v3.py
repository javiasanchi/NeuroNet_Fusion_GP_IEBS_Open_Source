import pandas as pd
import numpy as np
import os

ROOT = 'data/ida/ADNI'
BIO_ROOT1 = 'data/ida/ADNI/biospecimen_full'
BIO_ROOT2 = 'data/ida/ADNI/biospecimen_part2'

def get_rid(sid):
    try:
        return int(sid.split('_')[-1])
    except:
        return None

def clean_numeric(x):
    if isinstance(x, str):
        x = x.replace('>', '').replace('<', '').replace(',', '')
    try:
        return float(x)
    except:
        return np.nan

print("🔄 Iniciando consolidación V3 (Master Data + CSF + Plasma Multicentro)...")

# 1. Load Core Data
dx = pd.read_csv(os.path.join(ROOT, 'All_Subjects_DXSUM_16Feb2026.csv'))
scores = pd.read_csv(os.path.join(ROOT, 'All_Subjects_BLCHANGE_16Feb2026.csv'))
demog = pd.read_csv(os.path.join(ROOT, 'All_Subjects_PTDEMOG_16Feb2026.csv'))
entry = pd.read_csv(os.path.join(ROOT, 'All_Subjects_Study_Entry_16Feb2026.csv'))

# 2. Load Biomarkers
upenn_csf = pd.read_csv(os.path.join(BIO_ROOT1, 'UPENNBIOMK_MASTER_21Feb2026.csv'))
upenn_plasma = pd.read_csv(os.path.join(BIO_ROOT1, 'UPENNPLASMA_21Feb2026.csv'))

# New Plasma Data (Fujirebio & Selkoe)
fujirebio = pd.read_csv(os.path.join(BIO_ROOT2, 'FUJIREBIOABETAPLASMA_21Feb2026.csv'))
selkoe = pd.read_csv(os.path.join(BIO_ROOT2, 'SELKOELAB_NT1TAU_21Feb2026.csv'))

# 3. Processing Biomarkers
# CSF
for col in ['ABETA', 'TAU', 'PTAU']:
    upenn_csf[col] = upenn_csf[col].apply(clean_numeric)
upenn_csf = upenn_csf.sort_values(['RID', 'VISCODE']).drop_duplicates('RID')

# Plasma UPENN
upenn_plasma['AB42'] = upenn_plasma['AB42'].apply(clean_numeric)
upenn_plasma['AB40'] = upenn_plasma['AB40'].apply(clean_numeric)
upenn_plasma = upenn_plasma.sort_values(['RID', 'VISCODE']).drop_duplicates('RID')

# Plasma Fujirebio (Long format likely, based on 'ANALYTE' col)
fujirebio['CONCENTRATION'] = fujirebio['CONCENTRATION'].apply(clean_numeric)
fuji_42 = fujirebio[fujirebio['ANALYTE'] == 'AB142P'].copy().rename(columns={'CONCENTRATION': 'PLASMA_AB42_FUJI'})
fuji_40 = fujirebio[fujirebio['ANALYTE'] == 'AB140P'].copy().rename(columns={'CONCENTRATION': 'PLASMA_AB40_FUJI'})
fuji_42 = fuji_42.sort_values(['RID', 'VISCODE']).drop_duplicates('RID')
fuji_40 = fuji_40.sort_values(['RID', 'VISCODE']).drop_duplicates('RID')

# Plasma Selkoe Tau
selkoe['Dilution_Corrected_Value'] = selkoe['Dilution_Corrected_Value'].apply(clean_numeric)
selkoe = selkoe.rename(columns={'Dilution_Corrected_Value': 'PLASMA_TAU_SELKOE'})
selkoe = selkoe.dropna(subset=['RID']).sort_values(['RID', 'VISCODE2']).drop_duplicates('RID')

# 4. Master Merge
dx = dx.sort_values('update_stamp').drop_duplicates('RID', keep='last')
scores = scores.sort_values('update_stamp').drop_duplicates('RID', keep='last')
demog = demog.sort_values('update_stamp').drop_duplicates('RID', keep='last')
entry['RID'] = entry['subject_id'].apply(get_rid)
entry = entry.dropna(subset=['RID']).drop_duplicates('RID')

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
).merge(
    fuji_42[['RID', 'PLASMA_AB42_FUJI']], on='RID', how='left'
).merge(
    fuji_40[['RID', 'PLASMA_AB40_FUJI']], on='RID', how='left'
).merge(
    selkoe[['RID', 'PLASMA_TAU_SELKOE']], on='RID', how='left'
)

# 5. Engineering Ratios
df['CSF_AB_TAU_RATIO'] = df['ABETA'] / (df['TAU'] + 1e-8)
df['PLASMA_UPENN_RATIO'] = df['AB42'] / (df['AB40'] + 1e-8)
df['PLASMA_FUJI_RATIO'] = df['PLASMA_AB42_FUJI'] / (df['PLASMA_AB40_FUJI'] + 1e-8)

# Target
df['target'] = df['DIAGNOSIS'].fillna(df['DXNORM'].map({1:0})).fillna(df['DXMCI'].map({1:1})).fillna(df['DXAD'].map({1:2}))
df = df.dropna(subset=['target'])

# Save
output_path = 'Analytical_Biomarker_Project/data/consolidated_analytical_v3.csv'
df.to_csv(output_path, index=False)

print(f"✅ V3 con biomarcadores avanzados guardado en {output_path}")
print(f"📊 Muestras: {len(df)}")
print(f"🧪 Cobertura CSF: {df['ABETA'].notna().sum()}")
print(f"🧪 Cobertura Plasma Fuji: {df['PLASMA_FUJI_RATIO'].notna().sum()}")
print(f"🧪 Cobertura Plasma Tau (Selkoe): {df['PLASMA_TAU_SELKOE'].notna().sum()}")
print("\nClases:")
print(df['target'].value_counts())
