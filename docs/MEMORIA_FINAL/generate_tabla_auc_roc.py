"""
Genera la tabla de Resultados AUC-ROC — XGBoost Optimizado (§ 11.2.1)
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
BG_PERFECT = '#1a2a1a'   # Fila 1.0000 — verde oscuro

C_WHITE    = '#d4d4d4'
C_HEADER   = '#9cdcfe'
C_GREEN    = '#4ec9b0'
C_ORANGE   = '#ce9178'
C_GOLD     = '#ffd700'
C_RED      = '#f44747'
C_BLUE     = '#569cd6'
C_PURPLE   = '#c586c0'
C_GREY     = '#6a737d'
C_YELLOW   = '#dcdcaa'

# ── Datos ──────────────────────────────────────────────────────────────────────
#  clase              auc       interpretación                                bg         c_auc
rows = [
    ('NonDemented',      '0.9133', 'Alta discriminabilidad; confusión residual con VeryMild', BG_ROW_A,   C_WHITE),
    ('VeryMildDemented', '0.9275', 'Buena separación; casos límite con etapas adyacentes',  BG_ROW_B,   C_WHITE),
    ('MildDemented',     '0.9562', 'Muy alta discriminabilidad clínica',                   BG_ROW_A,   C_GREEN),
    ('ModerateDemented', '1.0000', '🏆 Separación perfecta — deterioro claramente distinguible', BG_PERFECT, C_GOLD),
]

headers = ['Clase', 'AUC-ROC', 'Interpretación']

# ── Anchos de columna ──────────────────────────────────────────────────────────
col_widths = [0.25, 0.15, 0.60]

# ── Dimensiones ────────────────────────────────────────────────────────────────
FIG_W   = 14
ROW_H   = 0.60
HROW_H  = 0.65
BAR_H   = 0.55
FOOTER  = 0.55
content_h = HROW_H + len(rows) * ROW_H + FOOTER
FIG_H   = BAR_H + 0.3 + content_h

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG_OUTER)

# ── Barra de título ────────────────────────────────────────────────────────────
bar_ax = fig.add_axes([0, 1 - BAR_H/FIG_H, 1, BAR_H/FIG_H])
bar_ax.set_facecolor(BG_OUTER)
bar_ax.set_xlim(0, 1); bar_ax.set_ylim(0, 1)
bar_ax.axis('off')

for cx, col in zip([0.04, 0.075, 0.11], ['#ff5f57','#ffbd2e','#28c840']):
    bar_ax.add_patch(plt.Circle((cx, 0.5), 0.016, color=col,
                                transform=bar_ax.transAxes, clip_on=False))

bar_ax.text(0.5, 0.5,
            '11.2.1  Rendimiento por Clase — Métricas AUC-ROC (One-vs-Rest)',
            color=C_WHITE, fontsize=11, fontfamily='monospace',
            ha='center', va='center', transform=bar_ax.transAxes)

# ── Área de contenido ──────────────────────────────────────────────────────────
mx = 0.03
c_y0 = 0.05 / FIG_H
c_h  = content_h / FIG_H

ax = fig.add_axes([mx, c_y0, 1 - 2*mx, c_h])
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
y_top    = content_h - 0.05
ax.add_patch(FancyBboxPatch((0, y_top - HROW_H), 1, HROW_H,
    boxstyle='square,pad=0', facecolor=BG_HEADER, edgecolor='none'))

for text, xc in zip(headers, col_c):
    ax.text(xc, y_top - HROW_H/2, text,
            color=C_HEADER, fontsize=10.5, fontfamily='monospace',
            fontweight='bold', ha='center', va='center')

ax.axhline(y_top - HROW_H, color='#3a3a5a', linewidth=1.2)

# ── Filas ──────────────────────────────────────────────────────────────────────
y_cur = y_top - HROW_H

for idx, (clase, auc, interp, bg, c_val) in enumerate(rows):
    ry = y_cur - ROW_H
    ax.add_patch(FancyBboxPatch((0, ry), 1, ROW_H,
        boxstyle='square,pad=0', facecolor=bg, edgecolor='none'))

    # Col 0: Clase
    ax.text(x_starts[0] + 0.01, ry + ROW_H/2, clase,
            color=C_BLUE, fontsize=10.5, fontfamily='monospace',
            ha='left', va='center')

    # Col 1: AUC
    ax.text(col_c[1], ry + ROW_H/2, auc,
            color=c_val, fontsize=11, fontfamily='monospace',
            fontweight='bold', ha='center', va='center')

    # Col 2: Interpretación
    ax.text(x_starts[2] + 0.01, ry + ROW_H/2, interp,
            color=C_WHITE if idx < 3 else C_GOLD, fontsize=10, fontfamily='monospace',
            ha='left', va='center')

    ax.axhline(ry, color='#2d2d4a', linewidth=0.6)
    y_cur = ry

# ── Marco ─────────────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0, y_cur), 1, y_top - y_cur,
    boxstyle='square,pad=0', linewidth=1.2,
    edgecolor='#3a3a5a', facecolor='none'))

# ── Nota al pie ────────────────────────────────────────────────────────────────
ax.text(0.012, y_cur - 0.25,
        '📈 AUC medio macro: 0.949  ·  El modelo se sitúa en la categoría de "Clasificador Excelente"',
        color=C_GREY, fontsize=9, fontfamily='monospace',
        ha='left', va='center')

# ── Guardado ───────────────────────────────────────────────────────────────────
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUT_DIR, '..', '..', 'reports', 'figures')

fname    = 'tabla_auc_roc_resultados.png'
out_path = os.path.join(OUT_DIR, fname)
fig_path = os.path.join(FIGURES_DIR, fname)

plt.savefig(out_path, dpi=180, bbox_inches='tight',
            facecolor=BG_OUTER, edgecolor='none')
plt.close()

shutil.copy2(out_path, fig_path)
print(f"✅  Guardado en: {out_path}")
print(f"✅  Copiado  en: {fig_path}")
