"""
Genera la tabla de Patrones Identificados en la búsqueda Optuna (§ 11.2)
como imagen PNG — estilo VS Code Dark+
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, shutil

# ── Paleta ─────────────────────────────────────────────────────────────────────
BG_OUTER  = '#1e1e2e'
BG_INNER  = '#1e1f2b'
BG_HEADER = '#252640'
BG_ROW_A  = '#22233a'
BG_ROW_B  = '#1e1f2b'

C_WHITE   = '#d4d4d4'
C_HEADER  = '#9cdcfe'
C_BLUE    = '#569cd6'
C_GREEN   = '#4ec9b0'
C_ORANGE  = '#ce9178'
C_YELLOW  = '#dcdcaa'
C_PURPLE  = '#c586c0'
C_TEAL    = '#4fc1ff'
C_GREY    = '#6a737d'
C_GOLD    = '#ffd700'

# ── Datos ──────────────────────────────────────────────────────────────────────
# parámetro          rango          interpretación                                bg        c_param   c_rango
rows = [
    ('n_estimators',    '650 – 920',    'Más árboles no mejoran pasado este umbral',      BG_ROW_A, C_BLUE,   C_TEAL),
    ('max_depth',       '5 – 7',        'Complejidad moderada; interacción con lr',        BG_ROW_B, C_BLUE,   C_TEAL),
    ('learning_rate',   '0.032 – 0.061','Ventana estrecha; muestreo log esencial',        BG_ROW_A, C_ORANGE, C_ORANGE),
    ('subsample',       '0.71 – 0.92',  'Bagging implícito óptimo por árbol',             BG_ROW_B, C_PURPLE, C_PURPLE),
    ('colsample_bytree','0.59 – 0.81',  'Selección parcial de features por árbol',        BG_ROW_A, C_PURPLE, C_PURPLE),
    ('reg_α  (L1)',     '0.009 – 0.062','L1 bajo — todas las features son relevantes',    BG_ROW_B, C_YELLOW, C_YELLOW),
    ('reg_λ  (L2)',     '0.84 – 3.62',  'L2 moderado — suavizado sin underfitting',       BG_ROW_A, C_YELLOW, C_YELLOW),
]

headers = ['Patrón identificado', 'Rango óptimo', 'Interpretación']

# col 0 → parámetro, col 1 → rango, col 2 → interpretación
col_widths = [0.22, 0.18, 0.58]

# ── Dimensiones ────────────────────────────────────────────────────────────────
FIG_W    = 14
ROW_H    = 0.58
HROW_H   = 0.65
BAR_H    = 0.55
FOOTER   = 0.42
n_rows   = len(rows)
content_h = HROW_H + n_rows * ROW_H + FOOTER
FIG_H    = BAR_H + 0.20 + content_h

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
            '11.2  Patrones identificados en el espacio de hiperparámetros Optuna TPE',
            color=C_WHITE, fontsize=11, fontfamily='monospace',
            ha='center', va='center', transform=bar_ax.transAxes)

# ── Área de contenido ──────────────────────────────────────────────────────────
mx  = 0.025
cy0 = 0.04 / FIG_H
ch  = content_h / FIG_H

ax = fig.add_axes([mx, cy0, 1 - 2*mx, ch])
ax.set_facecolor(BG_INNER)
ax.set_xlim(0, 1)
ax.set_ylim(0, content_h)
ax.axis('off')

# Posiciones x
x_starts = []
cur = 0.012
for w in col_widths:
    x_starts.append(cur)
    cur += w
col_c = [x_starts[i] + col_widths[i]/2 for i in range(len(col_widths))]

# ── Cabecera ───────────────────────────────────────────────────────────────────
y_top = content_h - 0.04
ax.add_patch(FancyBboxPatch((0, y_top - HROW_H), 1, HROW_H,
    boxstyle='square,pad=0', facecolor=BG_HEADER, edgecolor='none'))

for text, xc, align in zip(headers, col_c, ['center', 'center', 'center']):
    ax.text(xc, y_top - HROW_H/2, text,
            color=C_HEADER, fontsize=10.5, fontfamily='monospace',
            fontweight='bold', ha=align, va='center')

ax.axhline(y_top - HROW_H, color='#3a3a5a', linewidth=1.3)

# ── Filas ──────────────────────────────────────────────────────────────────────
y_cur = y_top - HROW_H

for idx, (param, rango, interp, bg, c_p, c_r) in enumerate(rows):
    ry = y_cur - ROW_H
    ax.add_patch(FancyBboxPatch((0, ry), 1, ROW_H,
        boxstyle='square,pad=0', facecolor=bg, edgecolor='none'))

    # Col 0 — parámetro (monospace con tick de código)
    ax.text(x_starts[0] + 0.010, ry + ROW_H/2, param,
            color=c_p, fontsize=10.5, fontfamily='monospace',
            ha='left', va='center', fontstyle='italic')

    # Col 1 — rango (centrado)
    ax.text(col_c[1], ry + ROW_H/2, rango,
            color=c_r, fontsize=10.2, fontfamily='monospace',
            fontweight='bold', ha='center', va='center')

    # Col 2 — interpretación (alineado izquierda)
    ax.text(x_starts[2] + 0.010, ry + ROW_H/2, interp,
            color=C_WHITE, fontsize=10, fontfamily='monospace',
            ha='left', va='center')

    ax.axhline(ry, color='#2d2d4a', linewidth=0.6)
    y_cur = ry

# Líneas verticales separadoras suaves
for xs in x_starts[1:]:
    ax.axvline(xs - 0.005, color='#2d2d4a', linewidth=0.8,
               ymin=(y_cur)/content_h, ymax=(y_top)/content_h)

# ── Marco ─────────────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0, y_cur), 1, y_top - y_cur,
    boxstyle='square,pad=0', linewidth=1.2,
    edgecolor='#3a3a5a', facecolor='none'))

# ── Nota al pie ────────────────────────────────────────────────────────────────
ax.text(0.012, y_cur - 0.24,
        '💡  Colores: azul = capacidad del árbol  ·  naranja = lr  ·  morado = regularización estocástica  ·  amarillo = L1/L2',
        color=C_GREY, fontsize=8.5, fontfamily='monospace', ha='left', va='center')

# ── Guardado ───────────────────────────────────────────────────────────────────
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUT_DIR, '..', '..', 'reports', 'figures')

fname    = 'tabla_patrones_optuna.png'
out_path = os.path.join(OUT_DIR, fname)
fig_path = os.path.join(FIGURES_DIR, fname)

plt.savefig(out_path, dpi=180, bbox_inches='tight',
            facecolor=BG_OUTER, edgecolor='none')
plt.close()

shutil.copy2(out_path, fig_path)
print(f"✅  Guardado en: {out_path}")
print(f"✅  Copiado  en: {fig_path}")
