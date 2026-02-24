import pandas as pd
import numpy as np
import os
import joblib
import json
import time
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Real-time stats for Phase 2
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_FILE = os.path.join(SCRIPT_DIR, '..', 'results', 'phase2_realtime_stats.json')

def save_phase2_stats(algo, best_acc, trial, total_trials, status):
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                data = json.load(f)
        except: data = {"results": {}}
    else:
        data = {"results": {}}
    
    if algo not in data['results']:
        data['results'][algo] = {"best_acc": 0, "trials": 0, "status": "Pending"}
    
    data['results'][algo] = {
        "best_acc": best_acc,
        "trials": trial,
        "total_trials": total_trials,
        "status": status
    }
    data['last_update'] = time.time()
    data['current_algo'] = algo
    
    with open(STATS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def run_phase2():
    print("💎 Fase 2: Optimización Hiperparamétrica (XGBoost, LightGBM, CatBoost) 💎\n")
    
    # Load Master Dataset
    data_path = 'Analytical_Biomarker_Project/data/master_combined_dataset.csv'
    df = pd.read_csv(data_path)
    df = df[df['BCMMSE'] >= 0]
    
    features = ['BCMMSE', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT', 'entry_age', 'APOE4_carrier', 'Hippocampus']
    X = df[features]
    y = df['target'].astype(int)
    
    # 70/30 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    
    TRIALS = 20 # Low for demo, increase for production

    # --- 1. XGBoost Optimization ---
    def objective_xgb(trial):
        params = {
            'n_estimators': 500,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'tree_method': 'hist',
            'random_state': 42
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Determine best so far manually if study not updated yet
        try:
            current_best = max(study_xgb.best_value, acc)
        except:
            current_best = acc
            
        save_phase2_stats("XGBoost", current_best, len(study_xgb.trials) + 1, TRIALS, "Optimizando...")
        return acc

    print("🚀 Optimizando XGBoost...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=TRIALS)
    save_phase2_stats("XGBoost", study_xgb.best_value, TRIALS, TRIALS, "Completado")

    # --- 2. LightGBM Optimization ---
    def objective_lgb(trial):
        params = {
            'n_estimators': 500,
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        try:
            current_best = max(study_lgb.best_value, acc)
        except:
            current_best = acc
            
        save_phase2_stats("LightGBM", current_best, len(study_lgb.trials) + 1, TRIALS, "Optimizando...")
        return acc

    print("🚀 Optimizando LightGBM...")
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=TRIALS)
    save_phase2_stats("LightGBM", study_lgb.best_value, TRIALS, TRIALS, "Completado")

    # --- 3. CatBoost Optimization ---
    def objective_cat(trial):
        params = {
            'iterations': 500,
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'random_seed': 42,
            'verbose': 0
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        try:
            current_best = max(study_cat.best_value, acc)
        except:
            current_best = acc
            
        save_phase2_stats("CatBoost", current_best, len(study_cat.trials) + 1, TRIALS, "Optimizando...")
        return acc

    print("🚀 Optimizando CatBoost...")
    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(objective_cat, n_trials=TRIALS)
    save_phase2_stats("CatBoost", study_cat.best_value, TRIALS, TRIALS, "Completado")

    # Final Summary
    print("\n" + "="*40)
    print("🏆 RESULTADOS FINALES - FASE 2")
    print("="*40)
    print(f"Mejor XGBoost:  {study_xgb.best_value:.4%}")
    print(f"Mejor LightGBM: {study_lgb.best_value:.4%}")
    print(f"Mejor CatBoost: {study_cat.best_value:.4%}")
    
    # Save the winners
    best_overall = max([('XGBoost', study_xgb), ('LightGBM', study_lgb), ('CatBoost', study_cat)], key=lambda x: x[1].best_value)
    print(f"\n🥇 EL CAMPEÓN ABSOLUTO ES: {best_overall[0]}")
    
    # Save best models
    joblib.dump(study_xgb.best_params, 'Analytical_Biomarker_Project/models/best_params_xgb.joblib')
    joblib.dump(study_lgb.best_params, 'Analytical_Biomarker_Project/models/best_params_lgb.joblib')
    joblib.dump(study_cat.best_params, 'Analytical_Biomarker_Project/models/best_params_cat.joblib')

if __name__ == "__main__":
    run_phase2()
