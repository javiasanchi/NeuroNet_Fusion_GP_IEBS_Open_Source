import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def train_baseline():
    data_path = 'Analytical_Biomarker_Project/data/consolidated_analytical.csv'
    if not os.path.exists(data_path):
        print("Data not found. Run consolidate_data.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Drop rows without target
    df = df.dropna(subset=['target'])
    
    # Features
    features = ['BCMMSE', 'BCADAS', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT', 'entry_age']
    X = df[features].fillna(df[features].median())
    y = df['target'].astype(int)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train
    print(f"Training Random Forest on {len(X_train)} samples...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- BASELINE ANALYTICAL MODEL RESULTS ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['CN', 'MCI', 'AD']))
    
    # Feature Importance
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(importances)

    # Save
    model_path = 'Analytical_Biomarker_Project/models/rf_baseline.joblib'
    joblib.dump(rf, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_baseline()
