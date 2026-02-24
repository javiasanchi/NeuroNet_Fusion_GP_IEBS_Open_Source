import pandas as pd
import numpy as np
import os
import glob
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

print("🚀 Iniciando Sprint de Mejora de Precisión...")

# 1. Cargar y Enriquecer Dataset
data_path = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/data/master_biomarker_dataset.csv'
df = pd.read_csv(data_path)

# --- Ingeniería de Features ---
df['MMSE_Age_Ratio'] = df['BCMMSE'] / (df['entry_age'] + 1)
df['Hippo_Age_Ratio'] = df['Hippocampus'] / (df['entry_age'] + 1)
df['has_CSF'] = df[['ABETA', 'TAU', 'PTAU']].notna().any(axis=1).astype(int)
df['has_PET'] = df['Centiloid'].notna().astype(int)

# Manejar valores -4.0 (NAs en ADNI)
df = df.replace(-4.0, np.nan)

features = ['BCMMSE', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT', 'entry_age', 
            'APOE4_carrier', 'Hippocampus', 'ABETA', 'TAU', 'PTAU', 'Centiloid',
            'MMSE_Age_Ratio', 'Hippo_Age_Ratio', 'has_CSF', 'has_PET']

X = df[features]
y = df['target']

# 2. Split Estratificado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Cálculo de Pesos para balancear MCI
weights = compute_sample_weight(class_weight='balanced', y=y_train)
# Aumentar peso extra a MCI (Clase 1) para reducir falsos negativos
weights[y_train == 1] *= 1.5 

# 4. Configuración del Modelo (Parámetros optimizados + Regularización fuerte)
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=5, # Evitar overfitting
    gamma=0.2,          # Penalización por complejidad
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,      # L1 regularización
    reg_lambda=2.0,     # L2 regularización
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# 5. Entrenamiento con pesos
print("⏳ Entrenando con pesos ajustados y nuevas features...")
model.fit(X_train, y_train, sample_weight=weights)

# 6. Evaluación
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✨ Nueva Precisión: {acc:.4f}")
print("\n📊 Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['CN', 'MCI', 'AD']))

print("\n🧩 Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# 7. Guardar Model
output_path = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/models/biomarker_xgb_sprint_v2.joblib'
joblib.dump(model, output_path)
print(f"\n💾 Modelo mejorado guardado en: {output_path}")

# 8. Importancia de Features
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n📊 Impacto de Nuevas Features:")
print(importances.head(10))
