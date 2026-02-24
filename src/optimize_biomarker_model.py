import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

print("🔍 Iniciando Optimización de Hiperparámetros para Dataset con Biomarcadores...")

# 1. Cargar Datos
data_path = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/data/master_biomarker_dataset.csv'
df = pd.read_csv(data_path)

features = ['BCMMSE', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT', 'entry_age', 
            'APOE4_carrier', 'Hippocampus', 'ABETA', 'TAU', 'PTAU', 'Centiloid']

X = df[features]
y = df['target']

# Análisis de balance
print("\n📊 Distribución de clases:")
print(y.value_counts(normalize=True))

# 2. Espacio de Búsqueda
param_dist = {
    'n_estimators': [500, 800, 1200],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5, 1],
    'min_child_weight': [1, 5, 10],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 5, 10]
}

# 3. Optimización
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

search = RandomizedSearchCV(
    xgb, 
    param_distributions=param_dist, 
    n_iter=30, 
    scoring='accuracy', 
    n_jobs=-1, 
    cv=5, 
    verbose=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n⏳ Ejecutando búsqueda aleatoria (30 iteraciones, 5-fold CV)...")
search.fit(X_train, y_train)

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"\n✨ Mejores Parámetros Encontrados:")
print(search.best_params_)

print(f"\n📈 Nueva Precisión en Test: {accuracy_score(y_test, y_pred):.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['CN', 'MCI', 'AD']))

# 4. Guardar Best Model
output_model_path = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/models/biomarker_xgb_optimized.joblib'
joblib.dump(best_model, output_model_path)
print(f"\n💾 Nuevo modelo optimizado guardado en: {output_model_path}")
