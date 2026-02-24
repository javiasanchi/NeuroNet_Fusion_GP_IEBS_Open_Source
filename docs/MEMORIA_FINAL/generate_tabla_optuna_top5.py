"""
Genera la tabla de resultados Optuna — Top-5 Trials XGBoost (§ 11.2)
como imagen PNG — estilo VS Code Dark+
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
BG_BEST    = '#1a2a1a'   # Fila ganadora — verde oscuro

C_WHITE    = '#d4d4d4'
C_HEADER   = '#9cdcfe'
C_GREEN    = '#4ec9b0'
C_ORANGE   = '#ce9178'
C_GOLD     = '#ffd700'
C_BLUE     = '#569cd6'
C_PURPLE   = '#c586c0'
C_YELLOW   = '#dcdcaa'
C_GREY     = '#6a737d'
C_RED      = '#f44747'
C_TEAL     = '#4fc1ff'

# ── Datos de los 5 trials ──────────────────────────────────────────────────────
# trial  n_est  depth  lr      subsample  colsample  reg_a    reg_l   acc_cv5   bg        c_acc  badge
rows = [
    ('1 🏆', '850',  '6', '0.048', '0.82', '0.74', '0.031', '2.14', '0.870', BG_BEST,  C_GOLD,   True),
    ('2',    '920',  '5', '0.041', '0.76', '0.68', '0.018', '3.62', '0.867', BG_ROW_A, C_GREEN,  False),
    ('3',    '780',  '7', '0.055', '0.88', '0.81', '0.045', '1.07', '0.863', BG_ROW_B, C_YELLOW, False),
    ('4',    '1100', '4', '0.032', '0.71', '0.59', '0.009', '5.30', '0.859', BG_ROW_A, C_ORANGE, False),
    ('5',    '650',  '6', '0.061', '0.92', '0.87', '0.062', '0.84', '0.855', BG_ROW_B, C_ORANGE, False),
]

headers = ['Trial', 'n_est', 'depth', 'lr', 'subsample', 'colsamp', 'reg_α', 'reg_λ', 'Acc CV-5']

# ── Anchos de columna ──────────────────────────────────────────────────────────
col_widths = [0.10, 0.09, 0.08, 0.09, 0.10, 0.10, 0.09, 0.09, 0.12]
# suma ≈ 0.86 → márgenes laterales cubren el resto

# ── Dimensiones ────────────────────────────────────────────────────────────────
FIG_W    = 15
ROW_H    = 0.60
HROW_H   = 0.65
BAR_H    = 0.55
SUBH     = 0.40   # subtítulo bajo barra
FOOTER   = 0.70
n_rows   = len(rows)
content_h = HROW_H + n_rows * ROW_H + FOOTER
FIG_H    = BAR_H + SUBH + 0.2 + content_h

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG_OUTER)

# ── Barra de título ────────────────────────────────────────────────────────────
bar_frac = BAR_H / FIG_H
bar_ax = fig.add_axes([0, 1 - bar_frac, 1, bar_frac])
bar_ax.set_facecolor(BG_OUTER)
bar_ax.set_xlim(0, 1); bar_ax.set_ylim(0, 1)
bar_ax.axis('off')

for cx, col in zip([0.04, 0.075, 0.11], ['#ff5f57', '#ffbd2e', '#28c840']):
    bar_ax.add_patch(plt.Circle((cx, 0.5), 0.017, color=col,
                                transform=bar_ax.transAxes, clip_on=False))

bar_ax.text(0.5, 0.5,
            '11.2  Optimización Optuna TPE — Top-5 Trials XGBoost (100 trials · CV-5)',
            color=C_WHITE, fontsize=11, fontfamily='monospace',
            ha='center', va='center', transform=bar_ax.transAxes)

# ── Subtítulo debajo de la barra ───────────────────────────────────────────────
sub_frac = SUBH / FIG_H
sub_ax = fig.add_axes([0, 1 - bar_frac - sub_frac, 1, sub_frac])
sub_ax.set_facecolor(BG_OUTER)
sub_ax.set_xlim(0, 1); sub_ax.set_ylim(0, 1)
sub_ax.axis('off')
sub_ax.text(0.5, 0.4,
            'Espacio de búsqueda: 9 hiperparámetros  ·  GPU: RTX 4070  ·  ~3 s / trial  ·  timeout: 3600 s',
            color=C_GREY, fontsize=9, fontfamily='monospace',
            ha='center', va='center', transform=sub_ax.transAxes)

# ── Área de contenido ──────────────────────────────────────────────────────────
mx   = 0.025
cy0  = 0.05 / FIG_H
ch   = content_h / FIG_H

ax = fig.add_axes([mx, cy0, 1 - 2*mx, ch])
ax.set_facecolor(BG_INNER)
ax.set_xlim(0, 1)
ax.set_ylim(0, content_h)
ax.axis('off')

# Posiciones x
x_starts = []
cx = 0.012
for w in col_widths:
    x_starts.append(cx)
    cx += w
col_c = [x_starts[i] + col_widths[i]/2 for i in range(len(col_widths))]

# ── Cabecera ───────────────────────────────────────────────────────────────────
y_top = content_h - 0.05
ax.add_patch(FancyBboxPatch((0, y_top - HROW_H), 1, HROW_H,
    boxstyle='square,pad=0', facecolor=BG_HEADER, edgecolor='none'))

for text, xc in zip(headers, col_c):
    ax.text(xc, y_top - HROW_H/2, text,
            color=C_HEADER, fontsize=10.5, fontfamily='monospace',
            fontweight='bold', ha='center', va='center')

ax.axhline(y_top - HROW_H, color='#3a3a5a', linewidth=1.3)

# ── Filas ──────────────────────────────────────────────────────────────────────
y_cur = y_top - HROW_H

for idx, (trial, n_est, depth, lr, subs, cols, rega, regl, acc, bg, c_acc, is_best) in enumerate(rows):
    ry = y_cur - ROW_H
    ax.add_patch(FancyBboxPatch((0, ry), 1, ROW_H,
        boxstyle='square,pad=0', facecolor=bg, edgecolor='none'))

    values = [trial, n_est, depth, lr, subs, cols, rega, regl, acc]
    colors = [
        C_GOLD if is_best else C_TEAL,   # trial
        C_BLUE,                           # n_est
        C_BLUE,                           # depth
        C_ORANGE,                         # lr
        C_PURPLE,                         # subsample
        C_PURPLE,                         # colsample
        C_YELLOW,                         # reg_alpha
        C_YELLOW,                         # reg_lambda
        c_acc,                            # accuracy
    ]
    weights = ['bold' if i in (0, 8) else 'normal' for i in range(len(values))]

    for i, (val, xc, clr, fw) in enumerate(zip(values, col_c, colors, weights)):
        ax.text(xc, ry + ROW_H/2, val,
                color=clr, fontsize=10.5 if i in (0, 8) else 10,
                fontfamily='monospace', fontweight=fw,
                ha='center', va='center')

    ax.axhline(ry, color='#2d2d4a', linewidth=0.6)
    y_cur = ry

# ── Marco ─────────────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0, y_cur), 1, y_top - y_cur,
    boxstyle='square,pad=0', linewidth=1.2,
    edgecolor='#3a3a5a', facecolor='none'))

# ── Notas al pie ───────────────────────────────────────────────────────────────
notes = [
    ('🏆  Trial 1 — mejor configuración: n_estimators=850, depth=6, lr=0.048  →  Best CV-5 Accuracy: 0.870', C_GOLD),
    ('    colsamp = colsample_bytree  ·  reg_α / reg_λ = regularización L1 / L2  ·  seed=42', C_GREY),
]
for i, (note, clr) in enumerate(notes):
    ax.text(0.012, y_cur - 0.15 - i * 0.26, note,
            color=clr, fontsize=8.6, fontfamily='monospace',
            ha='left', va='center')

# ── Guardado ───────────────────────────────────────────────────────────────────
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUT_DIR, '..', '..', 'reports', 'figures')

fname    = 'tabla_optuna_top5.png'
out_path = os.path.join(OUT_DIR, fname)
fig_path = os.path.join(FIGURES_DIR, fname)

plt.savefig(out_path, dpi=180, bbox_inches='tight',
            facecolor=BG_OUTER, edgecolor='none')
plt.close()

shutil.copy2(out_path, fig_path)
print(f"✅  Guardado en: {out_path}")
print(f"✅  Copiado  en: {fig_path}")
