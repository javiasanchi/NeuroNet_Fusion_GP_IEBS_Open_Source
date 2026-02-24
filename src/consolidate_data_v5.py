import pandas as pd
import numpy as np
import os
import pyreadr

ROOT = 'data/ida/ADNI'
BIO_ROOT1 = 'data/ida/ADNI/biospecimen_full'
BIO_ROOT2 = 'data/ida/ADNI/biospecimen_part2'
RDA_ROOT = 'data/ida/ADNI/ADNIMERGE2/data'
GEN_RAW = 'data/ida/ADNI/genetic_raw'
ASS_RAW = 'data/ida/ADNI/assessments_raw'

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

print("🔄 Iniciando Gran Consolidación V5 (Clínico Profundo + PRS + SOTA)...")

# 1. Basic Clinical & Labels
dx = pd.read_csv(os.path.join(ROOT, 'All_Subjects_DXSUM_16Feb2026.csv'))
scores = pd.read_csv(os.path.join(ROOT, 'All_Subjects_BLCHANGE_16Feb2026.csv'))
demog = pd.read_csv(os.path.join(ROOT, 'All_Subjects_PTDEMOG_16Feb2026.csv'))
entry = pd.read_csv(os.path.join(ROOT, 'All_Subjects_Study_Entry_16Feb2026.csv'))

# 2. Deep Phenotyping (Assessments Raw)
neurobat = pd.read_csv(os.path.join(ASS_RAW, 'NEUROBAT_21Feb2026.csv'))
# Select key cognitive tests: Trails A & B (Exe), BNT (Lang), Digital Span (Attention)
neurobat_feats = ['RID', 'TRAASCOR', 'TRABSCOR', 'BNTTOTAL', 'CATANIMSC', 'LDELTOTAL']
for col in neurobat_feats[1:]:
    neurobat[col] = neurobat[col].apply(clean_numeric)
neurobat = neurobat.sort_values(['RID', 'VISCODE']).drop_duplicates('RID')

# 3. Genetic PRS (Desikan Lab)
phs_df = pd.read_csv(os.path.join(GEN_RAW, 'DESIKANLAB_21Feb2026.csv'))
phs_df['PHS'] = phs_df['PHS'].apply(clean_numeric)
phs_df = phs_df.sort_values('update_stamp').drop_duplicates('RID', keep='last')

# 4. RDA Files (APOE, Volumes)
apoe_df = pyreadr.read_r(os.path.join(RDA_ROOT, 'APOERES.rda'))['APOERES']
vol_df = pyreadr.read_r(os.path.join(RDA_ROOT, 'UCSDVOL.rda'))['UCSDVOL']

apoe_df['APOE4_carrier'] = apoe_df['GENOTYPE'].apply(lambda x: 1 if pd.notna(x) and '4' in str(x) else 0)
apoe_df = apoe_df.sort_values('update_stamp').drop_duplicates('RID', keep='last')

vol_df['Hippocampus'] = vol_df['LHIPPOC'].apply(clean_numeric) + vol_df['RHIPPOC'].apply(clean_numeric)
vol_df['Entorhinal'] = vol_df['LENTORHIN'].apply(clean_numeric) + vol_df['RENTORHIN'].apply(clean_numeric)
vol_df['MidTemp'] = vol_df['LMIDTEMP'].apply(clean_numeric) + vol_df['RMIDTEMP'].apply(clean_numeric)
vol_df['Ventricles'] = vol_df['VENTRICLES'].apply(clean_numeric)
vol_df['ICV'] = vol_df['EICV'].apply(clean_numeric)
vol_df = vol_df.sort_values(['RID', 'EXAMDATE']).drop_duplicates('RID')

# 5. Fluid Biomarkers
upenn_csf = pd.read_csv(os.path.join(BIO_ROOT1, 'UPENNBIOMK_MASTER_21Feb2026.csv'))
upenn_plasma = pd.read_csv(os.path.join(BIO_ROOT1, 'UPENNPLASMA_21Feb2026.csv'))
for col in ['ABETA', 'TAU', 'PTAU']:
    upenn_csf[col] = upenn_csf[col].apply(clean_numeric)
upenn_csf = upenn_csf.sort_values(['RID', 'VISCODE']).drop_duplicates('RID')

# --- FINAL MERGE ---
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
    neurobat[neurobat_feats], on='RID', how='left'
).merge(
    phs_df[['RID', 'PHS']], on='RID', how='left'
).merge(
    apoe_df[['RID', 'APOE4_carrier']], on='RID', how='left'
).merge(
    vol_df[['RID', 'Hippocampus', 'Entorhinal', 'MidTemp', 'Ventricles', 'ICV']], on='RID', how='left'
).merge(
    upenn_csf[['RID', 'ABETA', 'TAU', 'PTAU']], on='RID', how='left'
)

# Engineering
df['CSF_AB_TAU_RATIO'] = df['ABETA'] / (df['TAU'] + 1e-8)
df['target'] = df['DIAGNOSIS'].fillna(df['DXNORM'].map({1:0})).fillna(df['DXMCI'].map({1:1})).fillna(df['DXAD'].map({1:2}))
df = df.dropna(subset=['target'])

# Save
output_path = 'Analytical_Biomarker_Project/data/consolidated_analytical_v5.csv'
df.to_csv(output_path, index=False)

print(f"✅ V5 (Dataset Profundo) guardado en {output_path}")
print(f"📊 Muestras: {len(df)}")
print(f"🧬 Cobertura PRS (PHS): {df['PHS'].notna().sum()}")
print(f"🧠 Cobertura Executive (Trails B): {df['TRABSCOR'].notna().sum()}")
print(f"🧬 Cobertura APOE: {df['APOE4_carrier'].notna().sum()}")
