import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
import os

def train_xgboost():
    data_path = 'Analytical_Biomarker_Project/data/consolidated_analytical.csv'
    if not os.path.exists(data_path):
        print("Data not found. Run consolidate_data.py first.")
        return

    df = pd.read_csv(data_path)
    df = df.dropna(subset=['target'])
    
    # Features
    features = ['BCMMSE', 'BCADAS', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT', 'entry_age']
    X = df[features].fillna(df[features].median())
    y = df['target'].astype(int)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # XGBoost requires labels starting from 0 (ADNI labels are 0,1,2 in my previous script, let's verify)
    # y.unique() was 1.0, 2.0, 3.0 in the previous check. Let's remap to 0,1,2
    y_train = y_train - 1
    y_test = y_test - 1

    print(f"Training XGBoost on {len(X_train)} samples...")
    
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        early_stopping_rounds=50
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- XGBOOST ANALYTICAL MODEL RESULTS ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['CN', 'MCI', 'AD']))
    
    # Save
    model_path = 'Analytical_Biomarker_Project/models/xgboost_baseline.joblib'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_xgboost()
