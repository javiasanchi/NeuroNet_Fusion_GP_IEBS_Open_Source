import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import xgboost as xgb

def analyze_importance():
    print("--- Analizando el peso e importancia de las nuevas variables normalizadas ---\n")
    
    data_path = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/data/master_biomarker_v2_normalized.csv'
    df = pd.read_csv(data_path)
    
    features = [
        'BCMMSE', 'BCCDR', 'BCFAQ',             # Cognitivos
        'PTGENDER', 'PTEDUCAT', 'entry_age',    # Demográficos
        'APOE4_carrier',                        # Genéticos
        'Hippo_Norm', 'Ento_Norm',              # Estructurales Normalizados
        'MidTemp_Norm', 'Vent_Norm',            # Nuevos Estructurales Normalizados
        'ABETA', 'TAU', 'PTAU'                  # Fluidos (CSF)
    ]
    
    df = df.dropna(subset=['target'])
    X = df[features]
    y = df['target'].astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # 1. CatBoost (Líder en el benchmark anterior)
    print("--- Importancia en CatBoost ---")
    cb_model = CatBoostClassifier(iterations=500, learning_rate=0.01, random_state=42, verbose=0)
    cb_model.fit(X_train, y_train)
    
    cb_importances = pd.Series(cb_model.get_feature_importance(), index=features).sort_values(ascending=False)
    print(cb_importances)
    
    # 2. XGBoost (Segundo en el benchmark)
    print("\n--- Importancia en XGBoost ---")
    xgb_model = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.01, random_state=42, tree_method='hist')
    xgb_model.fit(X_train, y_train)
    
    xgb_importances = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=False)
    print(xgb_importances)
    
    # Análisis específico de las nuevas variables
    new_vars = ['Hippo_Norm', 'Ento_Norm', 'MidTemp_Norm', 'Vent_Norm']
    print("\n--- Resumen de las nuevas variables normalizadas (CatBoost Weight) ---")
    summary_results = []
    for var in new_vars:
        weight = cb_importances[var]
        rank = int(cb_importances.index.get_loc(var) + 1)
        print(f"   * {var}: Peso {weight:.2f}% | Ranking: {rank}/{len(features)}")
        summary_results.append({'Variable': var, 'Weight': weight, 'Rank': rank})
    
    pd.DataFrame(summary_results).to_csv('E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/vars_importance_summary.csv', index=False)
    cb_importances.to_csv('E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/all_features_importance.csv')

if __name__ == "__main__":
    analyze_importance()
