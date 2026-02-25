"""
Genera la tabla de Justificación del Diseño (10.2.2)
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
C_PURPLE   = '#c586c0'
C_ORANGE   = '#ce9178'
C_GREEN    = '#4ec9b0'
C_YELLOW   = '#dcdcaa'
C_GREY     = '#6a737d'

# ── Datos de la Tabla ──────────────────────────────────────────────────────────
rows = [
    ('ResNet50 (50 capas)', 'ResNet18 (18 capas)', 'Mayor profundidad → mejor captura de atrofia sutil.', BG_ROW_A),
    ('DenseNet121',        'VGG-16',              'Conexiones densas → mejor flujo de gradientes en tejidos.', BG_ROW_B),
    ('LayerNorm',          'BatchNorm',           'LayerNorm mejor con batch sizes pequeños y datos heterogéneos.', BG_ROW_A),
    ('Dropout (0.5)',      'Dropout (0.3)',       'Regularización fuerte necesaria para vector de 3072 entradas.', BG_ROW_B),
    ('Label Smoothing 0.1','Sin smoothing',       'Gestiona la incertidumbre real en casos fronterizos MCI/AD.', BG_ROW_A),
]

headers = ['Decisión de Diseño', 'Alternativa Evaluada', 'Justificación Técnica']

# ── Anchos de columna ──────────────────────────────────────────────────────────
col_widths = [0.28, 0.28, 0.44]

# ── Dimensiones ────────────────────────────────────────────────────────────────
FIG_W   = 14
ROW_H   = 0.65
HROW_H  = 0.70
BAR_H   = 0.55
FOOTER  = 0.30
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

bar_ax.text(0.5, 0.5, '10.2.2  Justificación Técnica de Decisiones de Arquitectura',
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
    ax.text(xc, y_top - HROW_H/2, text, color=C_HEADER, fontsize=10.5, 
            fontfamily='monospace', fontweight='bold', ha='center', va='center')

ax.axhline(y_top - HROW_H, color='#3a3a5a', linewidth=1.2)

# ── Filas ──────────────────────────────────────────────────────────────────────
y_cur = y_top - HROW_H

for dec, alt, jus, bg in rows:
    ry = y_cur - ROW_H
    ax.add_patch(FancyBboxPatch((0, ry), 1, ROW_H,
                 boxstyle='square,pad=0', facecolor=bg, edgecolor='none'))

    # Decisión
    ax.text(x_starts[0] + 0.01, ry + ROW_H/2, dec, color=C_BLUE, 
            fontsize=10.5, fontfamily='monospace', fontweight='bold', ha='left', va='center')
    
    # Alternativa
    ax.text(x_starts[1] + 0.01, ry + ROW_H/2, alt, color=C_ORANGE, 
            fontsize=10.5, fontfamily='monospace', ha='left', va='center')
    
    # Justificación
    ax.text(x_starts[2] + 0.01, ry + ROW_H/2, jus, color=C_WHITE, 
            fontsize=10, fontfamily='monospace', ha='left', va='center')

    ax.axhline(ry, color='#2d2d4a', linewidth=0.6)
    y_cur = ry

# ── Marco ─────────────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0, y_cur), 1, y_top - y_cur,
             boxstyle='square,pad=0', linewidth=1.2, edgecolor='#3a3a5a', facecolor='none'))

# ── Guardado ───────────────────────────────────────────────────────────────────
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUT_DIR, '..', '..', 'reports', 'figures')

fname = 'tabla_10_2_2_justificacion.png'
out_path = os.path.join(OUT_DIR, fname)
fig_path = os.path.join(FIGURES_DIR, fname)

plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG_OUTER, edgecolor='none')
plt.close()

shutil.copy2(out_path, fig_path)
print(f"✅  Generada y copiada: {fname}")
