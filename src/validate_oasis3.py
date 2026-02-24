import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

def validate_oasis3():
    model_path = 'Analytical_Biomarker_Project/models/xgboost_sota_v5.joblib'
    data_path = 'Analytical_Biomarker_Project/data/consolidated_oasis3.csv'
    
    if not os.path.exists(model_path):
        print("Model not found. Train V5 first.")
        return
    if not os.path.exists(data_path):
        print("OASIS data not found. Run consolidate_oasis3.py first.")
        return

    # Load Model
    model = joblib.load(model_path)
    # Get the features the model was trained on
    features = [
        'BCMMSE', 'BCADAS', 'BCCDR', 'BCFAQ',
        'TRAASCOR', 'TRABSCOR', 'BNTTOTAL', 'CATANIMSC',
        'PTGENDER', 'PTEDUCAT', 'entry_age',
        'APOE4_carrier', 'PHS',
        'Hippocampus', 'Entorhinal',
        'ABETA', 'TAU', 'PTAU', 'CSF_AB_TAU_RATIO'
    ]

    # Load Data
    df = pd.read_csv(data_path)
    X = df[features]
    y = df['target'].astype(int)

    # Predict
    print(f"🔬 Validating model on {len(X)} samples from OASIS-3 (Zero-Shot)...")
    y_pred = model.predict(X)
    
    # Evaluate
    acc = accuracy_score(y, y_pred)
    
    print("\n" + "="*40)
    print("🌍 ZERO-SHOT VALIDATION: ADNI MODEL -> OASIS-3")
    print(f"Accuracy: {acc:.4%}")
    print("="*40)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['CN', 'MCI', 'AD']))

    # Feature Availability Report
    print("\nFeature Availability in OASIS-3:")
    for f in features:
        coverage = df[f].notna().mean() * 100
        print(f"  {f:20s}: {coverage:6.2f}%")

if __name__ == "__main__":
    validate_oasis3()
