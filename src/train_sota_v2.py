import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

def train_sota_analytical():
    data_path = 'Analytical_Biomarker_Project/data/consolidated_analytical_v2.csv'
    if not os.path.exists(data_path):
        print("Data not found. Run consolidate_data_v2.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Target remapping (0: CN, 1: MCI, 2: AD)
    y = (df['target'] - 1).astype(int)
    
    # Feature Selection (Clinical + Biomarkers)
    features = [
        'BCMMSE', 'BCADAS', 'BCCDR', 'BCFAQ',             # Cognitive/Functional
        'PTGENDER', 'PTEDUCAT', 'entry_age',              # Demographics
        'ABETA', 'TAU', 'PTAU', 'AB_TAU_RATIO',           # CSF Biomarkers
        'AB42', 'AB40', 'PLASMA_RATIO'                    # Plasma Biomarkers
    ]
    X = df[features]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print(f"Training SOTA XGBoost on {len(X_train)} samples with {len(features)} analytical features...")
    
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        early_stopping_rounds=50,
        tree_method='hist' # Efficient for missing values
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("SOTA ANALYTICAL MODEL (Biomarker-First)")
    print(f"Accuracy: {acc:.4%}")
    print("="*40)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['CN', 'MCI', 'AD']))
    
    # Feature Importance
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop Predictors (Feature Importance):")
    print(importances.head(10))

    # Save
    model_path = 'Analytical_Biomarker_Project/models/xgboost_sota_v2.joblib'
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to {model_path}")

if __name__ == "__main__":
    train_sota_analytical()
