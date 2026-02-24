import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
import os

def train_sota_v3():
    data_path = 'Analytical_Biomarker_Project/data/consolidated_analytical_v3.csv'
    if not os.path.exists(data_path):
        print("Data not found. Run consolidate_data_v3.py first.")
        return

    df = pd.read_csv(data_path)
    y = (df['target'] - 1).astype(int)
    
    # Comprehensive Feature List
    features = [
        'BCMMSE', 'BCADAS', 'BCCDR', 'BCFAQ',             # Cognitive/Functional
        'PTGENDER', 'PTEDUCAT', 'entry_age',              # Demographics
        'ABETA', 'TAU', 'PTAU', 'CSF_AB_TAU_RATIO',       # CSF
        'AB42', 'AB40', 'PLASMA_UPENN_RATIO',             # Plasma UPENN
        'PLASMA_AB42_FUJI', 'PLASMA_AB40_FUJI', 'PLASMA_FUJI_RATIO', # Plasma Fuji
        'PLASMA_TAU_SELKOE'                               # Plasma Tau Selkoe
    ]
    X = df[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print(f"Training SOTA XGBoost V3 on {len(X_train)} samples with {len(features)} multiset biomarkers...")
    
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
        tree_method='hist'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("SOTA V3: MULTI-BIOMARKER FUSION")
    print(f"Accuracy: {acc:.4%}")
    print("="*40)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['CN', 'MCI', 'AD']))
    
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop 10 Predictors:")
    print(importances.head(10))

    model_path = 'Analytical_Biomarker_Project/models/xgboost_sota_v3.joblib'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_sota_v3()
