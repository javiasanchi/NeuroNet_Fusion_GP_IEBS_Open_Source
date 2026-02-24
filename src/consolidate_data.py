import pandas as pd
import os

ROOT = 'data/ida/ADNI'

def get_rid(sid):
    try:
        return int(sid.split('_')[-1])
    except:
        return None

# Load Labels
dx = pd.read_csv(os.path.join(ROOT, 'All_Subjects_DXSUM_16Feb2026.csv'))
# Load Scores
scores = pd.read_csv(os.path.join(ROOT, 'All_Subjects_BLCHANGE_16Feb2026.csv'))
# Load Demographics
demog = pd.read_csv(os.path.join(ROOT, 'All_Subjects_PTDEMOG_16Feb2026.csv'))
# Load Age
entry = pd.read_csv(os.path.join(ROOT, 'All_Subjects_Study_Entry_16Feb2026.csv'))

# Merge on RID
# Note: DXSUM has diagnoses over time. We'll take the latest or most representative.
# For a baseline, let's take the latest per RID.
dx = dx.sort_values('update_stamp').drop_duplicates('RID', keep='last')
scores = scores.sort_values('update_stamp').drop_duplicates('RID', keep='last')
demog = demog.sort_values('update_stamp').drop_duplicates('RID', keep='last')

# entry has subject_id
entry['RID'] = entry['subject_id'].apply(get_rid)
entry = entry.dropna(subset=['RID']).drop_duplicates('RID')

# Merge
df = dx[['RID', 'DIAGNOSIS', 'DXNORM', 'DXMCI', 'DXAD']].merge(
    scores[['RID', 'BCMMSE', 'BCADAS', 'BCCDR', 'BCFAQ']], on='RID', how='inner'
).merge(
    demog[['RID', 'PTGENDER', 'PTEDUCAT']], on='RID', how='inner'
).merge(
    entry[['RID', 'entry_age']], on='RID', how='inner'
)

# Define Target
# 1=NL, 2=MCI, 3=AD (ADNI mapping usually)
df['target'] = df['DIAGNOSIS'].fillna(df['DXNORM'].map({1:0})).fillna(df['DXMCI'].map({1:1})).fillna(df['DXAD'].map({1:2}))

# Save
output_path = 'Analytical_Biomarker_Project/data/consolidated_analytical.csv'
df.to_csv(output_path, index=False)
print(f"Consolidated analytical data saved to {output_path}")
print(f"Total samples: {len(df)}")
print(df['target'].value_counts())
