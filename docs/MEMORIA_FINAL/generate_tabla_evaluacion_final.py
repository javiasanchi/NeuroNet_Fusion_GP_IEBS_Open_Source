"""
Genera la tabla de Evaluación Final — Matriz de Confusión (§ 11.1.4)
como imagen PNG — estilo VS Code Dark+, consistente con el resto del proyecto.
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
BG_PERFECT = '#1a2a1a'   # Fila 100 % — verde oscuro
BG_LOW     = '#2a2218'   # Fila más baja — naranja oscuro

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
#  clase              correctas  total  errores_desc                      prec      bg         c_prec
rows = [
    ('NonDemented',      '77',  '104',  '17 → VeryMild  ·  10 → Mild',  '74.0 %', BG_LOW,     C_ORANGE),
    ('VeryMildDemented', '78',   '95',  '10 → Non  ·  7 → Mild',        '82.1 %', BG_ROW_A,   C_YELLOW),
    ('MildDemented',     '78',   '88',  '8 → VeryMild  ·  2 → Non',     '88.6 %', BG_ROW_B,   C_GREEN),
    ('ModerateDemented', '113', '113',  '— sin errores',                 '100.0 %',BG_PERFECT, C_GOLD),
]

headers = ['Clase real', 'Correctas', 'Total test', 'Errores más frecuentes', 'Precisión']

# ── Anchos de columna (suma ≈ 1) ───────────────────────────────────────────────
col_widths = [0.22, 0.10, 0.10, 0.42, 0.13]

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
            '11.1.4  Evaluación Final — Precisión por Clase (Conjunto de Test)',
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
# Última columna: usar centro excepto para errores (alineado izq)

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

for idx, (clase, corr, total, errores, prec, bg, c_prec) in enumerate(rows):
    ry = y_cur - ROW_H
    ax.add_patch(FancyBboxPatch((0, ry), 1, ROW_H,
        boxstyle='square,pad=0', facecolor=bg, edgecolor='none'))

    # Col 0: Clase
    ax.text(x_starts[0] + 0.01, ry + ROW_H/2, clase,
            color=C_BLUE, fontsize=10.5, fontfamily='monospace',
            ha='left', va='center')

    # Col 1: Correctas
    ax.text(col_c[1], ry + ROW_H/2, corr,
            color=c_prec, fontsize=11, fontfamily='monospace',
            fontweight='bold', ha='center', va='center')

    # Col 2: Total
    ax.text(col_c[2], ry + ROW_H/2, total,
            color=C_WHITE, fontsize=10.5, fontfamily='monospace',
            ha='center', va='center')

    # Col 3: Errores (alinear izquierda)
    ax.text(x_starts[3] + 0.008, ry + ROW_H/2, errores,
            color=C_ORANGE if errores != '— sin errores' else C_GREY,
            fontsize=9.8, fontfamily='monospace',
            ha='left', va='center')

    # Col 4: Precisión
    ax.text(col_c[4], ry + ROW_H/2, prec,
            color=c_prec, fontsize=11, fontfamily='monospace',
            fontweight='bold', ha='center', va='center')

    ax.axhline(ry, color='#2d2d4a', linewidth=0.6)
    y_cur = ry

# ── Barra de precisión visual (sparkbar) ──────────────────────────────────────
bar_y = y_cur - 0.38
bar_x0 = x_starts[4]
bar_w  = col_widths[4] * 0.85
bar_full_w = bar_w

for idx, (_, _, _, _, prec, _, c_prec) in enumerate(rows):
    val = float(prec.replace(' %','')) / 100
    row_bar_y = (y_top - HROW_H) - (idx + 0.5) * ROW_H - ROW_H/4
    # mini barra proporcional detrás del número — solo decorativa

# ── Marco ─────────────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0, y_cur), 1, y_top - y_cur,
    boxstyle='square,pad=0', linewidth=1.2,
    edgecolor='#3a3a5a', facecolor='none'))

# ── Nota al pie ────────────────────────────────────────────────────────────────
note_lines = [
    '⚠  Los errores se producen entre etapas adyacentes (continuum patológico del Alzheimer)',
    '🏆 ModerateDemented: clasificación perfecta — patrones morfológicos altamente discriminativos',
]
for i, line in enumerate(note_lines):
    clr = C_ORANGE if i == 0 else C_GOLD
    ax.text(0.012, y_cur - 0.15 - i * 0.22, line,
            color=clr, fontsize=8.5, fontfamily='monospace',
            ha='left', va='center')

# ── Guardado ───────────────────────────────────────────────────────────────────
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUT_DIR, '..', '..', 'reports', 'figures')

fname    = 'tabla_evaluacion_final.png'
out_path = os.path.join(OUT_DIR, fname)
fig_path = os.path.join(FIGURES_DIR, fname)

plt.savefig(out_path, dpi=180, bbox_inches='tight',
            facecolor=BG_OUTER, edgecolor='none')
plt.close()

shutil.copy2(out_path, fig_path)
print(f"✅  Guardado en: {out_path}")
print(f"✅  Copiado  en: {fig_path}")
