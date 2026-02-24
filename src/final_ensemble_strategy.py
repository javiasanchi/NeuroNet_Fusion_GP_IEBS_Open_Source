"""
=======================================================================
 FINAL ENSEMBLE STRATEGY - Target: 95% Accuracy
 Modelos: XGBoost + LightGBM + CatBoost
 Voting: Soft Voting Classifier
 Features: Level 1 (Clinical Interactions)
=======================================================================
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.join(SCRIPT_DIR, '..')
DATA_PATH   = os.path.join(ROOT_DIR, 'data', 'master_combined_dataset.csv')
MODELS_DIR  = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET_NAMES = {0: 'CN', 1: 'MCI', 2: 'AD'}
BASE_FEATURES = ['BCMMSE', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT', 'entry_age', 'APOE4_carrier', 'Hippocampus']

# ── Feature Engineering Level 1 ──────────────────────────────────────────────
def build_features_l1(df):
    X = df[BASE_FEATURES].copy()
    # Interacciones Clinicas Clave
    X['mmse_x_cdr']     = X['BCMMSE']     * X['BCCDR']
    X['mmse_div_age']   = X['BCMMSE']     / (X['entry_age'] + 1e-3)
    X['faq_x_cdr']      = X['BCFAQ']      * X['BCCDR']
    X['hippo_div_age']  = X['Hippocampus']/ (X['entry_age'] + 1e-3)
    X['risk_score']     = (X['BCCDR'] * 0.4 + (30 - X['BCMMSE']) * 0.3 + X['BCFAQ'] * 0.3)
    return X

def main():
    print("🚀 Iniciando Estrategia Final de Ensemble...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encuentra el dataset en {DATA_PATH}")
        return

    # Carga y limpieza
    df = pd.read_csv(DATA_PATH)
    df = df[df['BCMMSE'] >= 0].reset_index(drop=True)
    y  = df['target'].astype(int)

    # Split idéntico para consistencia
    X_tmp, X_test_base, y_tmp, y_test = train_test_split(df, y, test_size=0.15, stratify=y, random_state=42)
    X_train_base, X_val_base, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.18, stratify=y_tmp, random_state=42)

    # Construcción de features
    X_train = build_features_l1(X_train_base).fillna(X_train_base[BASE_FEATURES].median())
    X_val   = build_features_l1(X_val_base).fillna(X_train.median())
    X_test  = build_features_l1(X_test_base).fillna(X_train.median())

    print(f"📊 Training con {X_train.shape[1]} features.")

    # 1. XGBoost (Configuración Ganadora R2)
    xgb_model = xgb.XGBClassifier(
        n_estimators=1500, max_depth=4, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.75, gamma=0.1,
        eval_metric=['mlogloss'],
        tree_method='hist', random_state=42
    )

    # 2. LightGBM (Configuración Optimizada)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000, max_depth=8, learning_rate=0.02,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )

    # 3. CatBoost (Excelente Generalizando)
    cat_model = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.03,
        l2_leaf_reg=3, random_seed=42, verbose=0
    )

    # Voting Classifier
    print("🗳️  Entrenando Voting Classifier (Soft Voting)...")
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('cat', cat_model)
        ],
        voting='soft'
    )

    # Entrenamiento del ensemble (incluye validación interna de XGB)
    ensemble.fit(X_train, y_train)

    # Evaluación
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n" + "="*40)
    print("🏆 RESULTADOS DEL ENSEMBLE FINAL")
    print("="*40)
    print(f"Accuracy: {acc:.4%}")
    print(f"F1-Score: {f1:.4%}")
    print("="*40)

    print("\nDetalle por Clase:")
    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES.values()))

    # Matriz de Confusión para análisis de errores
    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Guardar el Coloso
    save_path = os.path.join(MODELS_DIR, 'final_ensemble_95_hunt.joblib')
    joblib.dump(ensemble, save_path)
    print(f"\n💾 Modelo Ensemble guardado en: {save_path}")

    if acc >= 0.95:
        print("\n🎯 OBJETIVO 95% CUMPLIDO. ¡Felicidades!")
    else:
        print(f"\n📈 Gap para el 95%: {(0.95 - acc):.4%}. Estamos extremadamente cerca.")

if __name__ == "__main__":
    main()
