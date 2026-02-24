import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

print("🏆 Iniciando Estrategia de Ensemble para recuperar el 91% de precisión...")

# 1. Cargar y Limpiar
data_path = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/data/master_biomarker_dataset.csv'
df = pd.read_csv(data_path)

# Limpieza estricta de NAs marcados como -4.0
df = df.replace(-4.0, np.nan)

# Ingeniería de Features (Versión Refinada)
df['MMSE_Age'] = df['BCMMSE'] / (df['entry_age'] + 1)
df['Hippo_Age'] = df['Hippocampus'] / (df['entry_age'] + 1)
df['Bio_Score'] = df[['ABETA', 'TAU', 'PTAU', 'Centiloid']].notna().sum(axis=1)

features = ['BCMMSE', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT', 'entry_age', 
            'APOE4_carrier', 'Hippocampus', 'ABETA', 'TAU', 'PTAU', 'Centiloid',
            'MMSE_Age', 'Hippo_Age', 'Bio_Score']

X = df[features]
y = df['target']

# Imputación simple para modelos que no manejan NAs (RF)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
_, X_test_imp, _, y_test_imp = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# 2. Definir Modelos Base (Ajustados para precisión)
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss')
lgb = LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=64, subsample=0.8, random_state=42, verbose=-1)
rf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42)

# 3. Crear el Ensemble (Soft Voting para promediar probabilidades)
# Nota: Para el ensemble usaremos los datos imputados para que RF funcione correctamente
ensemble = VotingClassifier(
    estimators=[('xgb', xgb), ('lgb', lgb), ('rf', rf)],
    voting='soft'
)

print("⏳ Entrenando Ensemble (XGB + LGB + RF)...")
X_train_imp = imputer.transform(X_train) 
ensemble.fit(X_train_imp, y_train)

# 4. Evaluación
y_pred = ensemble.predict(X_test_imp)
acc = accuracy_score(y_test, y_pred)

print(f"\n✨ Precisión del Ensemble: {acc:.4f}")
print("\n📊 Reporte Final:")
print(classification_report(y_test, y_pred, target_names=['CN', 'MCI', 'AD']))

print("\n🧩 Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# 5. Guardar modelo y transformadores
model_dir = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/models'
joblib.dump(ensemble, f'{model_dir}/champion_ensemble_v1.joblib')
joblib.dump(imputer, f'{model_dir}/ensemble_imputer.joblib')

print(f"\n💾 Ensemble guardado como 'champion_ensemble_v1.joblib'")
