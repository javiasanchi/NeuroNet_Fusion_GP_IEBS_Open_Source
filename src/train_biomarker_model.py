import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import joblib
import os

print("🚀 Entrenando Modelo de Biomarcadores (XGBoost)...")

# 1. Load Dataset
data_path = 'Analytical_Biomarker_Project/data/master_biomarker_dataset.csv'
if not os.path.exists(data_path):
    data_path = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/data/master_biomarker_dataset.csv'

df = pd.read_csv(data_path)

# Handle missing data: XGBoost handles NaNs, but we should ensure categorical are encoded
# Gender is already 1/2. APOE4 is 0/1.

features = ['BCMMSE', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT', 'entry_age', 
            'APOE4_carrier', 'Hippocampus', 'ABETA', 'TAU', 'PTAU', 'Centiloid']
X = df[features]
y = df['target']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Model Definition (using params from champion if possible, or balanced defaults)
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# 4. Train
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n✅ Evaluación del Modelo:")
print(f"   - Accuracy: {acc:.4f}")
print(f"   - Weighted F1: {f1:.4f}")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['CN', 'MCI', 'AD']))

# 6. Save Model
model_dir = 'Analytical_Biomarker_Project/models'
os.makedirs(model_dir, exist_ok=True)
model_path = f'{model_dir}/biomarker_xgb_model.joblib'
joblib.dump(model, model_path)
print(f"\n💾 Modelo guardado en: {model_path}")

# 7. Feature Importance
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n📊 Importancia de Características:")
print(importances)
