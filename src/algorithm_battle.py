import pandas as pd
import numpy as np
import os
import joblib
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

STATS_FILE = 'Analytical_Biomarker_Project/results/battle_realtime_stats.json'

def save_stats(algo, acc, f1, status):
    # Try to load existing
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                data = json.load(f)
        except: data = {"results": []}
    else:
        data = {"results": []}
    
    # Check if algo already in list, if so update, else append
    updated = False
    for res in data['results']:
        if res['Algo'] == algo:
            res['Acc'] = acc
            res['F1'] = f1
            res['status'] = status
            updated = True
    if not updated:
        data['results'].append({'Algo': algo, 'Acc': acc, 'F1': f1, 'status': status})
    
    data['last_update'] = time.time()
    data['current_algo'] = algo
    data['global_status'] = "Training..." if status != "Complete" else "Phase 1 Done"

    with open(STATS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def run_algorithm_battle():
    print("⚔️  Iniciando Batalla de Algoritmos (ADNI + OASIS-3 Combined) ⚔️\n")
    
    # Reset stats
    if os.path.exists(STATS_FILE): os.remove(STATS_FILE)
    
    # Load Master Dataset
    data_path = 'Analytical_Biomarker_Project/data/master_combined_dataset.csv'
    df = pd.read_csv(data_path)
    
    # Cleaning
    df = df[df['BCMMSE'] >= 0]
    
    # Features & Target
    features = ['BCMMSE', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT', 'entry_age', 'APOE4_carrier', 'Hippocampus']
    X = df[features]
    y = df['target'].astype(int)
    
    # 70/30 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    
    X_train_filled = X_train.fillna(X_train.median())
    X_test_filled = X_test.fillna(X_test.median())

    # Algorithms list
    algos = [
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42), True),
        ('XGBoost', xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.01, random_state=42, tree_method='hist'), False),
        ('LightGBM', lgb.LGBMClassifier(n_estimators=500, learning_rate=0.01, random_state=42, verbose=-1), False),
        ('CatBoost', CatBoostClassifier(iterations=500, learning_rate=0.01, random_state=42, verbose=0), False),
        ('MLP-NN', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42), True)
    ]

    final_results = []
    for name, model, needs_filling in algos:
        print(f"⏳ Entrenando {name}...")
        save_stats(name, 0, 0, "En curso...")
        
        start_t = time.time()
        curr_X_train = X_train_filled if needs_filling else X_train
        curr_X_test = X_test_filled if needs_filling else X_test
        
        model.fit(curr_X_train, y_train)
        y_pred = model.predict(curr_X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        final_results.append({'Algo': name, 'Acc': acc, 'F1': f1})
        save_stats(name, acc, f1, "Finalizado")
        print(f"   Acc: {acc:.4f} | F1: {f1:.4f} ({time.time()-start_t:.2f}s)")

    # Final Summary
    res_df = pd.DataFrame(final_results).sort_values(by='Acc', ascending=False)
    save_stats("BATTLE_OVER", 0, 0, "Complete")
    
    print("\n" + "="*40)
    print("🏆 RESULTADOS FINALES DE LA BATALLA")
    print("="*40)
    print(res_df.to_string(index=False))
    
    res_df.to_csv('Analytical_Biomarker_Project/results/battle_results_v1.csv', index=False)

if __name__ == "__main__":
    run_algorithm_battle()
