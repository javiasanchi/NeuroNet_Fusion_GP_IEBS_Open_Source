import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

def run_test_30pct():
    print("🚀 Realizando test con un 30%% de registros sobre el Dataset Normalizado V2...\n")
    
    data_path = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/data/master_biomarker_v2_normalized.csv'
    if not os.path.exists(data_path):
        print(f"❌ Error: No se encuentra el archivo {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    # Definir características (incluyendo las nuevas normalizadas)
    features = [
        'BCMMSE', 'BCCDR', 'BCFAQ',             # Cognitivos
        'PTGENDER', 'PTEDUCAT', 'entry_age',    # Demográficos
        'APOE4_carrier',                        # Genéticos
        'Hippo_Norm', 'Ento_Norm',              # Estructurales Normalizados
        'MidTemp_Norm', 'Vent_Norm',            # Nuevos Estructurales Normalizados
        'ABETA', 'TAU', 'PTAU'                  # Fluidos (CSF)
    ]
    
    # Asegurarnos de que el target sea entero
    df = df.dropna(subset=['target'])
    X = df[features]
    y = df['target'].astype(int)
    
    # División 70/30 (30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    
    # Imputación simple para modelos que no manejan NaNs (RF)
    X_train_filled = X_train.fillna(X_train.median())
    X_test_filled = X_test.fillna(X_test.median())
    
    print(f"📊 Registros totales: {len(df)}")
    print(f"📈 Entrenamiento: {len(X_train)} | Test: {len(X_test)} (30%)\n")
    
    algos = [
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42), X_train_filled, X_test_filled),
        ('XGBoost', xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.01, random_state=42, tree_method='hist'), X_train, X_test),
        ('LightGBM', lgb.LGBMClassifier(n_estimators=500, learning_rate=0.01, random_state=42, verbose=-1), X_train, X_test),
        ('CatBoost', CatBoostClassifier(iterations=500, learning_rate=0.01, random_state=42, verbose=0), X_train, X_test)
    ]
    
    results = []
    
    for name, model, xt, xv in algos:
        print(f"⏳ Entrenando {name}...")
        model.fit(xt, y_train)
        y_pred = model.predict(xv)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({'Algo': name, 'Accuracy': acc, 'F1-Score': f1})
        print(f"   ✅ {name} -> Acc: {acc:.4f} | F1: {f1:.4f}")
    
    # Mostrar resumen final
    res_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
    print("\n" + "="*50)
    print("🏆 RESULTADOS FINALES (TEST 30%)")
    print("="*50)
    print(res_df.to_string(index=False))
    print("="*50)

    # Detalle del mejor modelo (probablemente XGBoost o CatBoost)
    best_algo_name = res_df.iloc[0]['Algo']
    print(f"\nReporte detallado para el mejor modelo ({best_algo_name}):")
    # Re-entrenar/predecir no es necesario si guardamos el objeto, pero por simplicidad:
    best_model = [m for n, m, xt, xv in algos if n == best_algo_name][0]
    best_xv = [xv for n, m, xt, xv in algos if n == best_algo_name][0]
    y_pred_best = best_model.predict(best_xv)
    print(classification_report(y_test, y_pred_best, target_names=['CN', 'MCI', 'AD']))

if __name__ == "__main__":
    run_test_30pct()
