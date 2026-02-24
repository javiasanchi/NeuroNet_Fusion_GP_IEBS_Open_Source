import pandas as pd
import numpy as np
import os
import pyreadr

ROOT = 'data/ida/ADNI'
BIO_ROOT1 = 'data/ida/ADNI/biospecimen_full'
BIO_ROOT2 = 'data/ida/ADNI/biospecimen_part2'
RDA_ROOT = 'data/ida/ADNI/ADNIMERGE2/data'

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

print("🔄 Iniciando gran consolidación V4 (Clínica + Fluidos + Genética + Volumétrica)...")

# 1. Load Core CSVs
dx = pd.read_csv(os.path.join(ROOT, 'All_Subjects_DXSUM_16Feb2026.csv'))
scores = pd.read_csv(os.path.join(ROOT, 'All_Subjects_BLCHANGE_16Feb2026.csv'))
demog = pd.read_csv(os.path.join(ROOT, 'All_Subjects_PTDEMOG_16Feb2026.csv'))
entry = pd.read_csv(os.path.join(ROOT, 'All_Subjects_Study_Entry_16Feb2026.csv'))

# 2. Load Fluid Biomarkers CSVs
upenn_csf = pd.read_csv(os.path.join(BIO_ROOT1, 'UPENNBIOMK_MASTER_21Feb2026.csv'))
upenn_plasma = pd.read_csv(os.path.join(BIO_ROOT1, 'UPENNPLASMA_21Feb2026.csv'))
fujirebio = pd.read_csv(os.path.join(BIO_ROOT2, 'FUJIREBIOABETAPLASMA_21Feb2026.csv'))
selkoe = pd.read_csv(os.path.join(BIO_ROOT2, 'SELKOELAB_NT1TAU_21Feb2026.csv'))

# 3. Load RDA files (Genetics and Volumes)
print("📥 Cargando archivos RDA (APOE, UCSD Volumes)...")
apoe_df = pyreadr.read_r(os.path.join(RDA_ROOT, 'APOERES.rda'))['APOERES']
vol_df = pyreadr.read_r(os.path.join(RDA_ROOT, 'UCSDVOL.rda'))['UCSDVOL']

# --- PROCESSING ---

# Genetics: Map APOE
apoe_df['APOE4_carrier'] = apoe_df['GENOTYPE'].apply(lambda x: 1 if pd.notna(x) and '4' in str(x) else 0)
apoe_df = apoe_df.sort_values('update_stamp').drop_duplicates('RID', keep='last')

# Volumes
vol_df['Hippocampus'] = vol_df['LHIPPOC'] + vol_df['RHIPPOC']
vol_df['Entorhinal'] = vol_df['LENTORHIN'] + vol_df['RENTORHIN']
vol_df = vol_df.sort_values(['RID', 'EXAMDATE']).drop_duplicates('RID', keep='first') # Take baseline volume

# Fluid CSF
for col in ['ABETA', 'TAU', 'PTAU']:
    upenn_csf[col] = upenn_csf[col].apply(clean_numeric)
upenn_csf = upenn_csf.sort_values(['RID', 'VISCODE']).drop_duplicates('RID')

# Fluid Plasma
upenn_plasma['AB42'] = upenn_plasma['AB42'].apply(clean_numeric)
upenn_plasma['AB40'] = upenn_plasma['AB40'].apply(clean_numeric)
upenn_plasma = upenn_plasma.sort_values(['RID', 'VISCODE']).drop_duplicates('RID')

fujirebio['CONCENTRATION'] = fujirebio['CONCENTRATION'].apply(clean_numeric)
fuji_42 = fujirebio[fujirebio['ANALYTE'] == 'AB142P'].copy().rename(columns={'CONCENTRATION': 'PLASMA_AB42_FUJI'})
fuji_40 = fujirebio[fujirebio['ANALYTE'] == 'AB140P'].copy().rename(columns={'CONCENTRATION': 'PLASMA_AB40_FUJI'})
fuji_42 = fuji_42.sort_values(['RID', 'VISCODE']).drop_duplicates('RID')
fuji_40 = fuji_40.sort_values(['RID', 'VISCODE']).drop_duplicates('RID')

selkoe['Dilution_Corrected_Value'] = selkoe['Dilution_Corrected_Value'].apply(clean_numeric)
selkoe = selkoe.rename(columns={'Dilution_Corrected_Value': 'PLASMA_TAU_SELKOE'})
selkoe = selkoe.dropna(subset=['RID']).sort_values(['RID', 'VISCODE2']).drop_duplicates('RID')

# --- MERGE ---
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
    apoe_df[['RID', 'APOE4_carrier']], on='RID', how='left'
).merge(
    vol_df[['RID', 'Hippocampus', 'Entorhinal', 'VENTRICLES', 'EICV']], on='RID', how='left'
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

# --- RATIOS & TARGET ---
df['CSF_AB_TAU_RATIO'] = df['ABETA'] / (df['TAU'] + 1e-8)
df['PLASMA_UPENN_RATIO'] = df['AB42'] / (df['AB40'] + 1e-8)
df['PLASMA_FUJI_RATIO'] = df['PLASMA_AB42_FUJI'] / (df['PLASMA_AB40_FUJI'] + 1e-8)

# Target
df['target'] = df['DIAGNOSIS'].fillna(df['DXNORM'].map({1:0})).fillna(df['DXMCI'].map({1:1})).fillna(df['DXAD'].map({1:2}))
df = df.dropna(subset=['target'])

# Save
output_path = 'Analytical_Biomarker_Project/data/consolidated_analytical_v4.csv'
df.to_csv(output_path, index=False)

print(f"✅ V4 (Dataset Final SOTA) guardado en {output_path}")
print(f"📊 Muestras Totales: {len(df)}")
print(f"🧬 Cobertura APOE: {df['APOE4_carrier'].notna().sum()}")
print(f"🧠 Cobertura Volumen Hipocampal: {df['Hippocampus'].notna().sum()}")
print(f"🧪 Cobertura Biomarcadores Fluido: {df['ABETA'].notna().sum()}")
print("\nClases:")
print(df['target'].value_counts())
