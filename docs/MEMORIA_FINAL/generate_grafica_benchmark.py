"""
Genera la gráfica de barras del Benchmark (§ 9.5)
como imagen PNG — estilo premium.
"""

import matplotlib.pyplot as plt
import os, shutil

# ── Paleta VS Code Dark+ ──────────────────────────────────────────────────────
BG_COLOR   = '#1e1e2e'
BAR_COLOR  = '#569cd6'
BEST_COLOR = '#4ec9b0'
TEXT_COLOR = '#d4d4d4'

data = {
    'XGBoost': 86.5,
    'LightGBM': 83.2,
    'CatBoost': 81.7,
    'LogReg': 66.0,
    'ResNet3D': 60.0,
    'MLP Large': 56.0,
    'SVM': 52.0,
    'MLP Small': 52.0,
    'HistGB': 48.0,
    'RandomForest': 44.0,
    'CNN3D Simple': 44.0,
    'K-Means': 36.0
}

# Invertir para que el mejor esté arriba
names = list(data.keys())[::-1]
values = list(data.values())[::-1]

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

bars = ax.barh(names, values, color=BAR_COLOR, height=0.7)
bars[-1].set_color(BEST_COLOR) # Resaltar el mejor

# Añadir valores al final de las barras
for bar in bars:
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width}%', 
            va='center', color=TEXT_COLOR, fontweight='bold', family='monospace')

# Estética
ax.set_title('9.5 Comparativa de Accuracy — Conjunto de Test', color=TEXT_COLOR, family='monospace', pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#45475a')
ax.spines['left'].set_color('#45475a')
ax.tick_params(axis='x', colors=TEXT_COLOR)
ax.tick_params(axis='y', colors=TEXT_COLOR)
ax.set_xlabel('Accuracy (%)', color=TEXT_COLOR, family='monospace')

plt.tight_layout()

# Guardar
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUT_DIR, '..', '..', 'reports', 'figures')
fname = 'grafica_benchmark_accuracy.png'
out_path = os.path.join(OUT_DIR, fname)
fig_path = os.path.join(FIGURES_DIR, fname)

plt.savefig(out_path, dpi=150, facecolor=BG_COLOR)
plt.close()

shutil.copy2(out_path, fig_path)
print(f"✅  Generada: {out_path}")
