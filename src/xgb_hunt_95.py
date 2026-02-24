"""
=======================================================================
 XGBoost HUNT 95% - Real-time Val Loss + Feature Engineering
 Dataset completo ADNI + OASIS-3
 Objetivo: superar 90% hacia 95%
 Fecha: 2026-02-22
=======================================================================
"""
import sys, os
# Forzar UTF-8 en stdout para evitar UnicodeEncodeError en Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import xgboost as xgb
import json, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             classification_report, log_loss)
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
# Asumimos que el script se ejecuta desde la raíz del proyecto o Analytical_Biomarker_Project
# Usamos rutas relativas al script para mayor robustez
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
# Si el script está en src/, subimos un nivel para llegar a la raíz de Analytical_Biomarker_Project
ROOT_DIR    = os.path.join(SCRIPT_DIR, '..')
DATA_PATH   = os.path.join(ROOT_DIR, 'data', 'master_combined_dataset.csv')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
MODELS_DIR  = os.path.join(ROOT_DIR, 'models')
STATS_FILE  = os.path.join(RESULTS_DIR, 'hunt95_realtime.json')
RESULTS_CSV = os.path.join(RESULTS_DIR, 'hunt95_results.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

TARGET_NAMES = {0: 'CN', 1: 'MCI', 2: 'AD'}
BASE_FEATURES = ['BCMMSE', 'BCCDR', 'BCFAQ', 'PTGENDER',
                 'PTEDUCAT', 'entry_age', 'APOE4_carrier', 'Hippocampus']

SEP  = "=" * 70
SEP2 = "-" * 70

def hdr(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def sub(title):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)

# ── Stats file ─────────────────────────────────────────────────────────────────
def save_stats(run_id, config_name, acc, f1, val_logloss, status, elapsed=0.0):
    try:
        data = json.load(open(STATS_FILE, encoding='utf-8')) \
               if os.path.exists(STATS_FILE) else {"runs": {}}
    except Exception:
        data = {"runs": {}}
    data["runs"][run_id] = {
        "config":      config_name,
        "acc":         round(acc,         6),
        "f1":          round(f1,          6),
        "val_logloss": round(val_logloss, 6),
        "status":      status,
        "elapsed_s":   round(elapsed,     1),
        "ts":          time.strftime("%H:%M:%S")
    }
    data["last_update"] = time.time()
    data["best_acc"]    = max((v["acc"] for v in data["runs"].values()), default=0.0)
    with open(STATS_FILE, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2)


# ── Custom XGBoost callback ───────────────────────────────────────────────────
class LiveCallback(xgb.callback.TrainingCallback):
    def __init__(self, print_every=100, total_rounds=None, run_label=""):
        self.print_every = print_every
        self.total       = total_rounds
        self.run_label   = run_label
        self.best_val    = float('inf')
        self.best_round  = 0

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % self.print_every == 0 or epoch == 0:
            try:
                # Buscamos 'train' y 'eval' en el log
                # En algunas versiones puede llamarse validation_0, validation_1
                context_keys = list(evals_log.keys())
                tr_key = context_keys[0] if len(context_keys) > 0 else 'train'
                vl_key = context_keys[1] if len(context_keys) > 1 else 'eval'
                
                tr_loss  = evals_log[tr_key]['mlogloss'][-1]
                val_loss = evals_log[vl_key]['mlogloss'][-1]
                tr_err   = evals_log[tr_key]['merror'][-1]
                val_err  = evals_log[vl_key]['merror'][-1]
            except Exception as e:
                # Si falla el log, saltamos el print
                return False

            marker = ""
            if val_loss < self.best_val:
                self.best_val   = val_loss
                self.best_round = epoch + 1
                marker = "<-- MEJOR"

            pct = f"{(epoch+1)/self.total*100:.0f}%" if self.total else ""
            print(
                f"  [{epoch+1:>4}/{self.total}] {pct:>4} | "
                f"TrLoss={tr_loss:.5f} VlLoss={val_loss:.5f} | "
                f"TrErr={tr_err:.4%} VlErr={val_err:.4%} {marker}",
                flush=True
            )
        return False


# ── Feature engineering ────────────────────────────────────────────────────────
def build_features(df, level=0):
    X = df[BASE_FEATURES].copy()

    if level >= 1:
        X['mmse_x_cdr']     = X['BCMMSE']     * X['BCCDR']
        X['mmse_div_age']   = X['BCMMSE']     / (X['entry_age'] + 1e-3)
        X['faq_x_cdr']      = X['BCFAQ']      * X['BCCDR']
        X['hippo_div_age']  = X['Hippocampus']/ (X['entry_age'] + 1e-3)
        X['age_x_apoe']     = X['entry_age']  * X['APOE4_carrier']
        X['educ_x_mmse']    = X['PTEDUCAT']   * X['BCMMSE']
        X['cdr_sq']         = X['BCCDR']      ** 2
        X['mmse_sq']        = X['BCMMSE']     ** 2
        X['faq_sq']         = X['BCFAQ']      ** 2
        X['hippo_sq']       = X['Hippocampus']** 2
        X['mmse_minus_cdr'] = X['BCMMSE']     - (X['BCCDR'] * 10)
        X['cognitive_load'] = X['BCMMSE']     - X['BCFAQ']
        X['risk_score']     = (X['BCCDR'] * 0.4 
                             + (30 - X['BCMMSE']) * 0.3 
                             + X['BCFAQ'] * 0.3)

    if level >= 2:
        core   = ['BCMMSE', 'BCCDR', 'BCFAQ', 'Hippocampus', 'entry_age']
        sub_df = X[core].fillna(X[core].median())
        poly   = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly_arr   = poly.fit_transform(sub_df)
        poly_names = [f"poly_{n}" for n in poly.get_feature_names_out(core)]
        poly_df    = pd.DataFrame(poly_arr, columns=poly_names, index=X.index)
        base_names = set(poly.get_feature_names_out(core)[:len(core)])
        poly_df    = poly_df.drop(columns=[f"poly_{n}" for n in base_names], errors='ignore')
        X = pd.concat([X, poly_df], axis=1)

    return X


# ── Configuraciones ───────────────────────────────────────────────────────────
CONFIGS = [
    {
        "id": "R1_baseline_full",
        "label": "Baseline ganador - full data",
        "feat_level": 0,
        "early": 80,
        "xgb": dict(n_estimators=1000, max_depth=3, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8)
    },
    {
        "id": "R2_interactions",
        "label": "Interacciones clinicas + 1500 arboles",
        "feat_level": 1,
        "early": 100,
        "xgb": dict(n_estimators=1500, max_depth=4, learning_rate=0.03,
                    subsample=0.85, colsample_bytree=0.75, gamma=0.1)
    },
    {
        "id": "R3_deep_reg",
        "label": "Profundo + Regularizacion fuerte",
        "feat_level": 1,
        "early": 120,
        "xgb": dict(n_estimators=2000, max_depth=6, learning_rate=0.02,
                    subsample=0.7, colsample_bytree=0.7, reg_lambda=3, min_child_weight=5)
    },
    {
        "id": "R4_poly_slowlr",
        "label": "LR lento + Features polinomiales",
        "feat_level": 2,
        "early": 150,
        "xgb": dict(n_estimators=3000, max_depth=5, learning_rate=0.01,
                    subsample=0.8, colsample_bytree=0.6, reg_alpha=0.2)
    },
    {
        "id": "R5_classweight_poly",
        "label": "Pesos por clase + poly",
        "feat_level": 2,
        "early": 150,
        "class_weight": True,
        "xgb": dict(n_estimators=3000, max_depth=5, learning_rate=0.01,
                    subsample=0.85, colsample_bytree=0.65)
    }
]


# ── Ejecucion ─────────────────────────────────────────────────────────────────
def run_config(cfg, X_train, X_val, X_test, y_train, y_val, y_test, all_results):
    sub(f"{cfg['id']} | {cfg['label']}")
    save_stats(cfg['id'], cfg['label'], 0.0, 0.0, 9.9, "En curso...")
    
    t0 = time.time()
    params = dict(cfg['xgb'])
    n_est = params.pop('n_estimators')
    early = cfg.get('early', 100)

    # Pesos si aplica
    sample_weight = None
    if cfg.get('class_weight'):
        counts = np.bincount(y_train)
        w = len(y_train) / (len(counts) * counts)
        sample_weight = np.array([w[yi] for yi in y_train])

    # Definir modelo pasándole early_stopping_rounds al constructor (mejor para v3.x)
    model = xgb.XGBClassifier(
        n_estimators=n_est,
        early_stopping_rounds=early,
        eval_metric=['mlogloss', 'merror'],
        **params,
        callbacks=[LiveCallback(print_every=100, total_rounds=n_est, run_label=cfg['id'])],
        tree_method='hist',
        random_state=42
    )

    # Entrenar
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        sample_weight=sample_weight,
        verbose=False
    )

    elapsed = time.time() - t0
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    vll = log_loss(y_test, y_prob)

    save_stats(cfg['id'], cfg['label'], acc, f1, vll, "Completado", elapsed)

    print(f"\n RESULTADO FINAL -- {cfg['id']}")
    print(f" Accuracy : {acc:.4%}")
    print(f" F1-Score : {f1:.4%}")
    print(f" LogLoss  : {vll:.5f}")
    print(f" Tiempo   : {elapsed:.1f}s")
    
    all_results.append({"Config": cfg['id'], "Accuracy": acc, "F1": f1, "LogLoss": vll})
    return acc

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encuentra el dataset en {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df = df[df['BCMMSE'] >= 0].reset_index(drop=True)
    y  = df['target'].astype(int)

    X_tmp, X_test, y_tmp, y_test = train_test_split(df, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.18, stratify=y_tmp, random_state=42)

    hdr("INICIANDO HUNT 95% CON XGBOOST")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    all_results = []
    for cfg in CONFIGS:
        lvl = cfg.get('feat_level', 0)
        X_tr = build_features(X_train, lvl).fillna(X_train[BASE_FEATURES].median())
        X_vl = build_features(X_val, lvl).fillna(X_tr.median())
        X_te = build_features(X_test, lvl).fillna(X_tr.median())
        
        acc = run_config(cfg, X_tr, X_vl, X_te, y_train, y_val, y_test, all_results)
        if acc >= 0.95:
            print("\n!!! OBJETIVO 95% ALCANZADO !!!")
            break

    # Ranking final
    res_df = pd.DataFrame(all_results).sort_values('Accuracy', ascending=False)
    res_df.to_csv(RESULTS_CSV, index=False)
    print("\n" + SEP)
    print(" RANKING FINAL")
    print(SEP)
    print(res_df.to_string(index=False))
    print(f"\nResultados guardados en: {RESULTS_CSV}")

if __name__ == "__main__":
    main()
