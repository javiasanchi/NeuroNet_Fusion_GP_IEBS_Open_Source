"""
Genera la tabla de métricas del Ensemble (10.4.1)
como imagen PNG — estilo VS Code Dark+.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, shutil

# ── Paleta VS Code Dark+ ──────────────────────────────────────────────────────
BG_OUTER   = '#1e1e2e'
BG_INNER   = '#1e1f2b'
BG_HEADER  = '#252640'
BG_ROW_A   = '#22233a'
BG_ROW_B   = '#1e1f2b'

C_WHITE    = '#d4d4d4'
C_HEADER   = '#9cdcfe'
C_BLUE     = '#569cd6'
C_ORANGE   = '#ce9178'
C_GREEN    = '#4ec9b0'
C_YELLOW   = '#dcdcaa'

# ── Datos de la Tabla ──────────────────────────────────────────────────────────
rows = [
    ('Accuracy',              '86.5 %',  '87.2 %',  BG_ROW_A),
    ('Varianza de predicción', '± 3.1 %', '± 1.8 %', BG_ROW_B),
    ('Robustez ante outliers', 'Media',   'Alta',     BG_ROW_A),
    ('Tiempo de inferencia',   '34 ms',   '89 ms',    BG_ROW_B),
]

headers = ['Métrica de Rendimiento', 'XGBoost Individual', 'Ensemble (XGB+LGB+CAT)']

# ── Configuración Visual ──────────────────────────────────────────────────────
FIG_W   = 12
ROW_H   = 0.65
HROW_H  = 0.70
BAR_H   = 0.55
content_h = HROW_H + len(rows) * ROW_H + 0.2
FIG_H   = BAR_H + content_h

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG_OUTER)

# Barra de título
bar_ax = fig.add_axes([0, 1 - BAR_H/FIG_H, 1, BAR_H/FIG_H])
bar_ax.set_facecolor(BG_OUTER)
bar_ax.axis('off')
bar_ax.text(0.5, 0.5, '10.4.1  Comparativa: Modelo Individual vs. Ensemble de Producción',
            color=C_WHITE, fontsize=11, fontfamily='monospace', ha='center', va='center')

# Contenido
ax = fig.add_axes([0.05, 0.05 / FIG_H, 0.9, content_h / FIG_H])
ax.set_facecolor(BG_INNER)
ax.set_xlim(0, 1); ax.set_ylim(0, content_h)
ax.axis('off')

col_widths = [0.4, 0.3, 0.3]
x_starts = [0.02, 0.42, 0.72]
col_c = [x + w/2 for x, w in zip(x_starts, col_widths)]

# Cabecera
y_top = content_h - 0.05
ax.add_patch(FancyBboxPatch((0, y_top - HROW_H), 1, HROW_H, boxstyle='square,pad=0', facecolor=BG_HEADER))
for text, xc in zip(headers, col_c):
    ax.text(xc, y_top - HROW_H/2, text, color=C_HEADER, fontsize=10.5, fontfamily='monospace', fontweight='bold', ha='center', va='center')

# Filas
y_cur = y_top - HROW_H
for met, ind, ens, bg in rows:
    ry = y_cur - ROW_H
    ax.add_patch(FancyBboxPatch((0, ry), 1, ROW_H, boxstyle='square,pad=0', facecolor=bg))
    ax.text(x_starts[0], ry + ROW_H/2, met, color=C_WHITE, fontsize=10.5, fontfamily='monospace', ha='left', va='center')
    ax.text(col_c[1], ry + ROW_H/2, ind, color=C_ORANGE, fontsize=10.5, fontfamily='monospace', ha='center', va='center')
    ax.text(col_c[2], ry + ROW_H/2, ens, color=C_GREEN, fontsize=10.5, fontfamily='monospace', fontweight='bold', ha='center', va='center')
    y_cur = ry

# Guardado
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
fname = 'tabla_10_4_1_comparativa_ensemble.png'
plt.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight', facecolor=BG_OUTER)
shutil.copy2(os.path.join(OUT_DIR, fname), os.path.join(OUT_DIR, '..', '..', 'reports', 'figures', fname))
print(f"✅  Generada: {fname}")
