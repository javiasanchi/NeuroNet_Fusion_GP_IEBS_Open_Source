"""
Genera la tabla de Resultados del Benchmark (§ 9.3)
como imagen PNG — estilo VS Code Dark+.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, shutil

# ── Paleta ─────────────────────────────────────────────────────────────────────
BG_OUTER   = '#1e1e2e'
BG_INNER   = '#1e1f2b'
BG_HEADER  = '#252640'
BG_ROW_A   = '#22233a'
BG_ROW_B   = '#1e1f2b'
BG_BEST    = '#1a2a1a'

C_WHITE    = '#d4d4d4'
C_HEADER   = '#9cdcfe'
C_BLUE     = '#569cd6'
C_ORANGE   = '#ce9178'
C_GOLD     = '#ffd700'
C_GREEN    = '#4ec9b0'
C_GREY     = '#6a737d'

# ── Datos ──────────────────────────────────────────────────────────────────────
#  Modelo                     Acc     F1-Score  AUC-ROC  Tiempo  bg
rows = [
    ('K-Means (No Supervisado)', '36.0%', '—',      '—',     '2.1',  BG_ROW_A),
    ('Random Forest',           '44.0%', '0.41',   '0.68',  '8.4',  BG_ROW_B),
    ('SimpleCNN-3D',            '44.0%', '0.42',   '0.71',  '380',  BG_ROW_A),
    ('HistGradientBoosting',    '48.0%', '0.46',   '0.72',  '12.7', BG_ROW_B),
    ('MLP Small',               '52.0%', '0.50',   '0.76',  '45',   BG_ROW_A),
    ('SVM (Best)',              '52.0%', '0.70',   '0.90',  '28.3', BG_ROW_B),
    ('MLP Large',               '56.0%', '0.54',   '0.79',  '127',  BG_ROW_A),
    ('ResNet3D',                '60.0%', '0.59',   '0.82',  '1.840',BG_ROW_B),
    ('Logistic Regression',     '66.0%', '0.66',   '0.86',  '1.2',  BG_ROW_A),
    ('CatBoost',                '81.7%', '0.81',   '0.86',  '52',   BG_ROW_B),
    ('LightGBM',                '83.2%', '0.83',   '0.88',  '18',   BG_ROW_A),
    ('XGBoost (Champion) 🏆',    '86.5%', '0.86',   '0.89',  '34',   BG_BEST),
]

headers = ['Modelo', 'Accuracy', 'F1-Score', 'AUC-ROC', 'Tiempo (s)']

# ── Anchos de columna ──────────────────────────────────────────────────────────
col_widths = [0.36, 0.16, 0.16, 0.16, 0.16]

# ── Dimensiones ────────────────────────────────────────────────────────────────
FIG_W   = 14
ROW_H   = 0.55
HROW_H  = 0.60
BAR_H   = 0.55
FOOTER  = 0.35
content_h = HROW_H + len(rows) * ROW_H + FOOTER
FIG_H   = BAR_H + 0.2 + content_h

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG_OUTER)

# ── Barra de título ────────────────────────────────────────────────────────────
bar_ax = fig.add_axes([0, 1 - BAR_H/FIG_H, 1, BAR_H/FIG_H])
bar_ax.set_facecolor(BG_OUTER)
bar_ax.set_xlim(0, 1); bar_ax.set_ylim(0, 1)
bar_ax.axis('off')

for cx, col in zip([0.04, 0.075, 0.11], ['#ff5f57','#ffbd2e','#28c840']):
    bar_ax.add_patch(plt.Circle((cx, 0.5), 0.016, color=col, transform=bar_ax.transAxes))

bar_ax.text(0.5, 0.5, '9.3  Resultados Comparativos del Benchmark',
            color=C_WHITE, fontsize=11, fontfamily='monospace',
            ha='center', va='center', transform=bar_ax.transAxes)

# ── Área de contenido ──────────────────────────────────────────────────────────
mx = 0.03
c_y0 = 0.04 / FIG_H
c_h  = content_h / FIG_H

ax = fig.add_axes([mx, c_y0, 1 - 2*mx, c_h])
ax.set_facecolor(BG_INNER)
ax.set_xlim(0, 1); ax.set_ylim(0, content_h)
ax.axis('off')

# Posiciones x
x_starts = []
cx = 0.01
for w in col_widths:
    x_starts.append(cx)
    cx += w
col_c = [x_starts[i] + col_widths[i]/2 for i in range(len(col_widths))]

# ── Cabecera ───────────────────────────────────────────────────────────────────
y_top = content_h - 0.05
ax.add_patch(FancyBboxPatch((0, y_top - HROW_H), 1, HROW_H,
             boxstyle='square,pad=0', facecolor=BG_HEADER, edgecolor='none'))

for text, xc in zip(headers, col_c):
    ax.text(xc, y_top - HROW_H/2, text, color=C_HEADER, fontsize=10.5, 
            fontfamily='monospace', fontweight='bold', ha='center', va='center')

ax.axhline(y_top - HROW_H, color='#3a3a5a', linewidth=1.2)

# ── Filas ──────────────────────────────────────────────────────────────────────
y_cur = y_top - HROW_H

for model, acc, f1, auc, time, bg in rows:
    ry = y_cur - ROW_H
    ax.add_patch(FancyBboxPatch((0, ry), 1, ROW_H,
                 boxstyle='square,pad=0', facecolor=bg, edgecolor='none'))

    # Modelo
    color_mod = C_GOLD if 'Champion' in model else C_BLUE
    ax.text(x_starts[0] + 0.01, ry + ROW_H/2, model, color=color_mod, 
            fontsize=10.5, fontfamily='monospace', fontweight='bold' if 'Champion' in model else 'normal',
            ha='left', va='center')
    
    # Métricas
    for i, val in enumerate([acc, f1, auc, time]):
        color_val = C_GREEN if 'Champion' in model else C_WHITE
        ax.text(col_c[i+1], ry + ROW_H/2, val, color=color_val, 
                fontsize=10.5, fontfamily='monospace', ha='center', va='center')

    ax.axhline(ry, color='#2d2d4a', linewidth=0.6)
    y_cur = ry

# ── Marco ─────────────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0, y_cur), 1, y_top - y_cur,
             boxstyle='square,pad=0', linewidth=1.2, edgecolor='#3a3a5a', facecolor='none'))

# ── Guardado ───────────────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUT_DIR, '..', '..', 'reports', 'figures')

fname = 'tabla_benchmark_resultados.png'
out_path = os.path.join(OUT_DIR, fname)
fig_path = os.path.join(FIGURES_DIR, fname)

plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG_OUTER, edgecolor='none')
plt.close()

shutil.copy2(out_path, fig_path)
print(f"✅  Guardado en: {out_path}")
print(f"✅  Copiado  en: {fig_path}")
