import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

def train_final_model():
    print("🏆 Iniciando Entrenamiento Final del Modelo Ganador (XGBoost) 🏆")
    
    # Load Master Dataset
    data_path = 'Analytical_Biomarker_Project/data/master_combined_dataset.csv'
    df = pd.read_csv(data_path)
    df = df[df['BCMMSE'] >= 0] # Filtro de calidad
    
    features = ['BCMMSE', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT', 'entry_age', 'APOE4_carrier', 'Hippocampus']
    X = df[features]
    y = df['target'].astype(int)
    
    # Load Best Params
    best_params = joblib.load('Analytical_Biomarker_Project/models/best_params_xgb.joblib')
    # Actualizamos n_estimators y añadimos early stopping
    best_params['n_estimators'] = 2000 
    best_params['early_stopping_rounds'] = 100
    best_params['eval_metric'] = ['mlogloss', 'merror']
    best_params['tree_method'] = 'hist'
    
    # Split for validation tracking
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    
    model = xgb.XGBClassifier(**best_params)
    
    print(f"📊 Entrenando con {len(X_train)} muestras y validando con {len(X_val)}...")
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )
    
    # Results
    results = model.evals_result()
    epochs = len(results['validation_0']['mlogloss'])
    x_axis = range(0, epochs)
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    plt.plot(x_axis, results['validation_1']['mlogloss'], label='Validation')
    plt.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    
    # Plot Error
    plt.subplot(1, 2, 2)
    plt.plot(x_axis, results['validation_0']['merror'], label='Train')
    plt.plot(x_axis, results['validation_1']['merror'], label='Validation')
    plt.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    
    plot_path = 'Analytical_Biomarker_Project/results/final_model_loss.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"📈 Gráfica de pérdida guardada en: {plot_path}")
    
    # Final Evaluation
    y_pred = model.predict(X_val)
    print("\n" + "="*40)
    print("✅ EVALUACIÓN FINAL (Hold-out 10%)")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4%}")
    print("="*40)
    
    # Check unique labels to avoid ValueError
    unique_labels = sorted(np.unique(np.concatenate([y_val, y_pred])))
    target_map = {0: 'CN', 1: 'MCI', 2: 'AD'}
    target_names = [target_map[l] for l in unique_labels]
    
    print(classification_report(y_val, y_pred, target_names=target_names))
    
    # Save Model
    model_save_path = 'Analytical_Biomarker_Project/models/final_model_alzheimer_sota.joblib'
    joblib.dump(model, model_save_path)
    print(f"💾 Modelo final guardado en: {model_save_path}")

if __name__ == "__main__":
    train_final_model()
