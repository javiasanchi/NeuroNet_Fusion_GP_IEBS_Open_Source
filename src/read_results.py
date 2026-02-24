import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

model_path = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/models/biomarker_xgb_sprint_v2.joblib'
data_path = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/data/master_biomarker_dataset.csv'

df = pd.read_csv(data_path)
df = df.replace(-4.0, np.nan)
df['MMSE_Age_Ratio'] = df['BCMMSE'] / (df['entry_age'] + 1)
df['Hippo_Age_Ratio'] = df['Hippocampus'] / (df['entry_age'] + 1)
df['has_CSF'] = df[['ABETA', 'TAU', 'PTAU']].notna().any(axis=1).astype(int)
df['has_PET'] = df['Centiloid'].notna().astype(int)

features = ['BCMMSE', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT', 'entry_age', 
            'APOE4_carrier', 'Hippocampus', 'ABETA', 'TAU', 'PTAU', 'Centiloid',
            'MMSE_Age_Ratio', 'Hippo_Age_Ratio', 'has_CSF', 'has_PET']

X = df[features]
y = df['target']

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = joblib.load(model_path)
y_pred = model.predict(X_test)

print(f"Precisión: {accuracy_score(y_test, y_pred):.4f}")
print("\nReporte:")
print(classification_report(y_test, y_pred, target_names=['CN', 'MCI', 'AD']))
print("\nMatriz:")
print(confusion_matrix(y_test, y_pred))
