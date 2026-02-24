"""
========================================================
 BENCHMARK 50% DATA — Multi-Parametrization Test
 Modelos: LightGBM, XGBoost, CatBoost
 Dataset: 50% aleatorio estratificado del Master Dataset
 Fecha: 2026-02-21
========================================================
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import json
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report
)
import warnings
warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, '..', 'data',    'master_combined_dataset.csv')
RESULTS_DIR  = os.path.join(SCRIPT_DIR, '..', 'results')
MODELS_DIR   = os.path.join(SCRIPT_DIR, '..', 'models')
STATS_FILE   = os.path.join(RESULTS_DIR, 'benchmark_50pct_stats.json')
RESULTS_CSV  = os.path.join(RESULTS_DIR, 'benchmark_50pct_results.csv')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

FEATURES = ['BCMMSE', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT',
            'entry_age', 'APOE4_carrier', 'Hippocampus']
TARGET   = 'target'
TARGET_NAMES = ['CN', 'MCI', 'AD']

# ─── Real-time stats helper ───────────────────────────────────────────────────
def save_stats(model_name: str, config_id: str, acc: float, f1: float,
               status: str, elapsed: float = 0.0):
    try:
        data = json.load(open(STATS_FILE)) if os.path.exists(STATS_FILE) else {"results": {}}
    except Exception:
        data = {"results": {}}

    key = f"{model_name}_{config_id}"
    data["results"][key] = {
        "model":   model_name,
        "config":  config_id,
        "acc":     round(acc,  6),
        "f1":      round(f1,   6),
        "status":  status,
        "elapsed": round(elapsed, 2)
    }
    data["last_update"]  = time.time()
    data["current_task"] = key
    with open(STATS_FILE, 'w') as fh:
        json.dump(data, fh, indent=2)


# ─── Parametrizaciones ────────────────────────────────────────────────────────
LGBM_CONFIGS = {
    "LGB_A_shallow_fast": dict(
        n_estimators=300,  max_depth=4, num_leaves=31,
        learning_rate=0.05, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=20, lambda_l1=0.0, lambda_l2=0.0,
        random_state=42, verbose=-1
    ),
    "LGB_B_deep_slow": dict(
        n_estimators=700,  max_depth=10, num_leaves=127,
        learning_rate=0.01, feature_fraction=0.7,
        bagging_fraction=0.7, bagging_freq=1,
        min_child_samples=10, lambda_l1=0.1, lambda_l2=0.1,
        random_state=42, verbose=-1
    ),
    "LGB_C_regularized": dict(
        n_estimators=500,  max_depth=6, num_leaves=63,
        learning_rate=0.02, feature_fraction=0.6,
        bagging_fraction=0.6, bagging_freq=5,
        min_child_samples=30, lambda_l1=0.5, lambda_l2=0.5,
        random_state=42, verbose=-1
    ),
    "LGB_D_balanced": dict(
        n_estimators=500,  max_depth=7, num_leaves=80,
        learning_rate=0.03, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=3,
        min_child_samples=15, lambda_l1=0.05, lambda_l2=0.05,
        random_state=42, verbose=-1
    ),
}

XGB_CONFIGS = {
    "XGB_A_shallow_fast": dict(
        n_estimators=300,  max_depth=3,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, gamma=0,
        reg_alpha=0, reg_lambda=1,
        tree_method='hist', random_state=42
    ),
    "XGB_B_deep_regularized": dict(
        n_estimators=700,  max_depth=8,
        learning_rate=0.01, subsample=0.7,
        colsample_bytree=0.7, gamma=0.2,
        reg_alpha=0.1, reg_lambda=2,
        tree_method='hist', random_state=42
    ),
    "XGB_C_dart": dict(
        n_estimators=500,  max_depth=6,
        learning_rate=0.02, subsample=0.9,
        colsample_bytree=0.9, booster='dart',
        rate_drop=0.1, skip_drop=0.5,
        tree_method='hist', random_state=42
    ),
    "XGB_D_conservative": dict(
        n_estimators=500,  max_depth=4,
        learning_rate=0.03, subsample=0.6,
        colsample_bytree=0.6, gamma=0.5,
        reg_alpha=0.5, reg_lambda=5,
        min_child_weight=10,
        tree_method='hist', random_state=42
    ),
}

CAT_CONFIGS = {
    "CAT_A_fast": dict(
        iterations=300,  depth=4,
        learning_rate=0.05, l2_leaf_reg=3,
        border_count=32, random_strength=1,
        bagging_temperature=0,
        random_seed=42, verbose=0
    ),
    "CAT_B_deep": dict(
        iterations=700,  depth=8,
        learning_rate=0.01, l2_leaf_reg=5,
        border_count=128, random_strength=2,
        bagging_temperature=0.5,
        random_seed=42, verbose=0
    ),
    "CAT_C_regularized": dict(
        iterations=500,  depth=6,
        learning_rate=0.02, l2_leaf_reg=10,
        border_count=64, random_strength=1.5,
        bagging_temperature=1.0,
        random_seed=42, verbose=0
    ),
    "CAT_D_balanced": dict(
        iterations=500,  depth=5,
        learning_rate=0.03, l2_leaf_reg=7,
        border_count=64, random_strength=1,
        bagging_temperature=0.3,
        random_seed=42, verbose=0
    ),
}


# ─── Main benchmark ───────────────────────────────────────────────────────────
def run_benchmark():
    print("=" * 65)
    print("  🔬 BENCHMARK 50% DATA — Multi-Parametrization Test")
    print("  Modelos: LightGBM · XGBoost · CatBoost")
    print("=" * 65)

    # ── Load & sample ──────────────────────────────────────────────────────────
    df_full = pd.read_csv(DATA_PATH)
    df_full = df_full[df_full['BCMMSE'] >= 0]

    df_50, _ = train_test_split(
        df_full, test_size=0.50, stratify=df_full[TARGET], random_state=42
    )
    print(f"\n📂 Dataset completo : {len(df_full):,} filas")
    print(f"📂 Submuestra 50%   : {len(df_50):,}  filas")
    print(f"📊 Distribución de la submuestra:")
    dist = df_50[TARGET].value_counts().sort_index()
    for cls, cnt in dist.items():
        print(f"   • Clase {int(cls)} ({TARGET_NAMES[int(cls)]}): {cnt:,} muestras")

    X = df_50[FEATURES]
    y = df_50[TARGET].astype(int)

    # 70/30 train/test split interno
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    print(f"\n🔀 Split interno 70/30:")
    print(f"   • Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print()

    all_results = []

    # ── Helper: evaluate one config ───────────────────────────────────────────
    def evaluate_config(model_name, config_name, model, x_tr, x_te):
        tag = f"[{model_name}] {config_name}"
        print(f"  ⏳ {tag}...")
        save_stats(model_name, config_name, 0.0, 0.0, "En curso...")
        t0 = time.time()

        model.fit(x_tr, y_train)
        y_pred = model.predict(x_te)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        elapsed = time.time() - t0

        save_stats(model_name, config_name, acc, f1, "Completado", elapsed)
        print(f"     ✅ Acc={acc:.4%}  F1={f1:.4%}  Prec={prec:.4%}  Rec={rec:.4%}  ({elapsed:.1f}s)")

        return {
            "Model":      model_name,
            "Config":     config_name,
            "Accuracy":   acc,
            "F1":         f1,
            "Precision":  prec,
            "Recall":     rec,
            "Time_s":     round(elapsed, 2)
        }

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print("─" * 65)
    print("🌿 LightGBM")
    print("─" * 65)
    for cfg_name, params in LGBM_CONFIGS.items():
        model = lgb.LGBMClassifier(**params)
        res   = evaluate_config("LightGBM", cfg_name, model, X_train, X_test)
        all_results.append(res)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("─" * 65)
    print("🚀 XGBoost")
    print("─" * 65)
    for cfg_name, params in XGB_CONFIGS.items():
        model = xgb.XGBClassifier(**params)
        res   = evaluate_config("XGBoost", cfg_name, model, X_train, X_test)
        all_results.append(res)

    # ── CatBoost ──────────────────────────────────────────────────────────────
    print("─" * 65)
    print("🐱 CatBoost")
    print("─" * 65)
    for cfg_name, params in CAT_CONFIGS.items():
        model = CatBoostClassifier(**params)
        res   = evaluate_config("CatBoost", cfg_name, model, X_train, X_test)
        all_results.append(res)

    # ── Summary ───────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    results_df.to_csv(RESULTS_CSV, index=False)

    print("\n" + "=" * 65)
    print("🏆  RANKING FINAL — 50% Data / Multi-Parametrization")
    print("=" * 65)
    print(f"\n{'Pos':>3}  {'Model':<12} {'Config':<30} {'Acc':>8}  {'F1':>8}  {'Time':>6}")
    print("─" * 65)
    for i, row in results_df.iterrows():
        medal = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "  "))
        print(f"{medal} {i+1:>2}  {row['Model']:<12} {row['Config']:<30} "
              f"{row['Accuracy']:.4%}  {row['F1']:.4%}  {row['Time_s']:>5.1f}s")

    print("\n" + "─" * 65)
    best = results_df.iloc[0]
    print(f"🏆 GANADOR ABSOLUTO : {best['Model']} — {best['Config']}")
    print(f"   Accuracy  : {best['Accuracy']:.4%}")
    print(f"   F1-Score  : {best['F1']:.4%}")
    print(f"   Precision : {best['Precision']:.4%}")
    print(f"   Recall    : {best['Recall']:.4%}")
    print("─" * 65)

    # Per-model best
    print("\n📋 MEJOR CONFIGURACIÓN POR MODELO:")
    for mdl in ["LightGBM", "XGBoost", "CatBoost"]:
        sub = results_df[results_df["Model"] == mdl].iloc[0]
        print(f"  {mdl:<12} → {sub['Config']:<32} Acc={sub['Accuracy']:.4%}  F1={sub['F1']:.4%}")

    print(f"\n💾 Resultados guardados en: {RESULTS_CSV}")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()
