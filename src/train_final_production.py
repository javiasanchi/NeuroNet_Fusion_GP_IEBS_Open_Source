import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight

def train_final_production_model():
    print("--- Iniciando Entrenamiento Final de Producción (80/20 Split) ---")
    
    data_path = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/data/master_biomarker_v2_normalized.csv'
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
    
    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # Cálculo de Pesos de Clase para balancear el aprendizaje
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, weights))
    
    # Mapear pesos a cada muestra en el set de entrenamiento
    sample_weights = np.array([class_weights_dict[cls] for cls in y_train])
    
    print(f"Pesos de clase calculados: {class_weights_dict}")
    
    # Hiperparámetros Optimizados (obtenidos en el GridSearch)
    model = xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        tree_method='hist'
    )
    
    print("⏳ Entrenando modelo final con 80% de los datos...")
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluación Final sobre el 20% reservado
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n--- RESULTADOS FINALES DEL MODELO DE PRODUCCIÓN ---")
    print(f"Total Registros: {len(df)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score (Weighted): {f1:.4f}")
    print("\nReporte de Clasificación (20% Test Set):")
    print(classification_report(y_test, y_pred, target_names=['CN', 'MCI', 'AD']))
    
    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    
    # Guardar Modelo y Metadatos
    model_path = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/models/neuro_fusion_final_v1.joblib'
    joblib.dump({
        'model': model,
        'features': features,
        'metrics': {'accuracy': acc, 'f1': f1},
        'params': model.get_params()
    }, model_path)
    
    # Exportar resultados para documentación
    res_summary = {
        'Metric': ['Accuracy', 'F1-Score', 'Train Samples', 'Test Samples'],
        'Value': [acc, f1, len(X_train), len(X_test)]
    }
    pd.DataFrame(res_summary).to_csv('E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/results/final_model_stats.csv', index=False)
    
    print(f"\n✅ Modelo guardado exitosamente en: {model_path}")

if __name__ == "__main__":
    train_final_production_model()
