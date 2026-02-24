import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb

def optimize_xgboost():
    print("--- Iniciando optimización de hiperparámetros para XGBoost ---")
    
    data_path = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/data/master_biomarker_v2_normalized.csv'
    if not os.path.exists(data_path):
        print(f"Error: No se encuentra el archivo {data_path}")
        return
    
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['target'])
    
    features = [
        'BCMMSE', 'BCCDR', 'BCFAQ', 
        'PTGENDER', 'PTEDUCAT', 'entry_age', 
        'APOE4_carrier', 
        'Hippo_Norm', 'Ento_Norm', 
        'MidTemp_Norm', 'Vent_Norm', 
        'ABETA', 'TAU', 'PTAU'
    ]
    
    X = df[features]
    y = df['target'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    
    # Definición de la malla de parámetros
    param_grid = {
        'n_estimators': [500, 800],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    
    xgb_base = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        tree_method='hist',
        eval_metric='mlogloss'
    )
    
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    print("⏳ Ejecutando GridSearch (esto puede tardar unos minutos)...")
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    print("\n✅ Mejores hiperparámetros encontrados:")
    print(best_params)
    
    # Evaluación con el mejor modelo
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n--- Rendimiento del Modelo Optimizado ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score (Weighted): {f1:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=['CN', 'MCI', 'AD']))
    
    # Guardar el modelo
    model_dir = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, 'xgboost_optimized_v2.joblib')
    joblib.dump(best_model, model_path)
    print(f"\n📂 Modelo optimizado guardado en: {model_path}")
    
    # Actualizar importancia de variables con el modelo optimizado
    importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
    importances.to_csv('E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/results/optimized_xgb_importances.csv')

if __name__ == "__main__":
    optimize_xgboost()
