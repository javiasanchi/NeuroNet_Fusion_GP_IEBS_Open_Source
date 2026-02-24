"""
Genera la tabla de Curva de Aprendizaje como imagen PNG
Estilo: VS Code Dark+ — consistente con los bloques de código del proyecto
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os, shutil

# ── Paleta de colores VS Code Dark+ ───────────────────────────────────────────
BG_OUTER   = '#1e1e2e'   # Fondo exterior (barra de título)
BG_INNER   = '#1e1f2b'   # Fondo del contenido
BG_HEADER  = '#252640'   # Fondo cabecera tabla
BG_ROW_A   = '#22233a'   # Fila par
BG_ROW_B   = '#1e1f2b'   # Fila impar
BG_PHASE1  = '#2a2a1a'   # Fila hito Fase 1
BG_BEST    = '#1a2a1a'   # Fila best checkpoint
BG_STOP    = '#2a1a1a'   # Fila early stopping

C_WHITE    = '#d4d4d4'
C_YELLOW   = '#dcdcaa'
C_GREEN    = '#4ec9b0'
C_ORANGE   = '#ce9178'
C_RED      = '#f44747'
C_BLUE     = '#569cd6'
C_GOLD     = '#ffd700'
C_GREY     = '#6a737d'
C_HEADER   = '#9cdcfe'
C_TITLE    = '#c586c0'

# ── Datos de la tabla ──────────────────────────────────────────────────────────
rows = [
    # época  train_loss  val_loss  train_acc  val_acc  hito                          bg         color_hito
    ('1',    '2.142',    '2.089',  '31.2 %',  '33.8 %', 'Inicio — pesos aleatorios',  BG_ROW_A,  C_GREY),
    ('10',   '1.654',    '1.712',  '55.4 %',  '52.1 %', 'Convergencia del clasificador', BG_ROW_B, C_WHITE),
    ('20',   '1.287',    '1.398',  '68.9 %',  '64.7 %', '⬛ Fin Fase 1 — backbone descongelado', BG_PHASE1, C_YELLOW),
    ('30',   '0.981',    '1.045',  '74.2 %',  '71.3 %', 'Fine-tuning activo',         BG_ROW_A,  C_WHITE),
    ('50',   '0.723',    '0.841',  '80.8 %',  '78.2 %', 'Cruce del 80 % Val Acc',    BG_ROW_B,  C_WHITE),
    ('70',   '0.542',    '0.698',  '85.1 %',  '82.9 %', 'Estabilización loss',        BG_ROW_A,  C_WHITE),
    ('85',   '0.487',    '0.631',  '87.4 %',  '86.5 %', '🏆 Best checkpoint — guardado', BG_BEST, C_GOLD),
    ('95',   '0.461',    '0.659',  '88.2 %',  '85.9 %', '⚠️  Early Stopping (Val Loss ↑)', BG_STOP, C_RED),
]

headers = ['Época', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc', 'Hito']

# ── Proporciones de columnas (suma 1) ─────────────────────────────────────────
col_widths = [0.07, 0.11, 0.10, 0.10, 0.10, 0.52]

# ── Dimensiones figura ─────────────────────────────────────────────────────────
FIG_W   = 16
ROW_H   = 0.55          # altura por fila de datos
HROW_H  = 0.60          # altura cabecera
VPAD    = 1.1            # padding vertical total
BAR_H   = 0.55          # barra de título

n_rows  = len(rows)
content_h = HROW_H + n_rows * ROW_H + 0.35   # + nota al pie
FIG_H   = BAR_H + VPAD + content_h

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG_OUTER)

# ── Barra de título (macOS style) ─────────────────────────────────────────────
bar_ax = fig.add_axes([0, 1 - BAR_H/FIG_H, 1, BAR_H/FIG_H])
bar_ax.set_facecolor(BG_OUTER)
bar_ax.set_xlim(0, 1); bar_ax.set_ylim(0, 1)
bar_ax.axis('off')

for i, (cx, col) in enumerate(zip([0.05, 0.09, 0.13], ['#ff5f57','#ffbd2e','#28c840'])):
    bar_ax.add_patch(plt.Circle((cx, 0.5), 0.015, color=col, transform=bar_ax.transAxes))

bar_ax.text(0.5, 0.5,
            '11.1.3  Curva de Aprendizaje — Evolución del Entrenamiento CNN Dual-Backbone',
            color=C_WHITE, fontsize=11, fontfamily='monospace',
            ha='center', va='center', transform=bar_ax.transAxes)

# ── Área de contenido ──────────────────────────────────────────────────────────
margin_x = 0.03
content_y0 = 0.05 / FIG_H
content_h_frac = content_h / FIG_H

ax = fig.add_axes([margin_x, content_y0, 1 - 2*margin_x, content_h_frac])
ax.set_facecolor(BG_INNER)
ax.set_xlim(0, 1)
ax.set_ylim(0, content_h)
ax.axis('off')

# Calcular posiciones x de columnas
x_starts = []
cx = 0.015
for w in col_widths:
    x_starts.append(cx)
    cx += w
col_centers = [x_starts[i] + col_widths[i]/2 for i in range(len(col_widths))]

# ── Cabecera ───────────────────────────────────────────────────────────────────
y_top = content_h - 0.05
y = y_top - HROW_H/2

ax.add_patch(FancyBboxPatch((0, y_top - HROW_H), 1, HROW_H,
    boxstyle='square,pad=0', facecolor=BG_HEADER, edgecolor='none'))

for i, (text, xc) in enumerate(zip(headers, col_centers)):
    ax.text(xc, y_top - HROW_H/2, text,
            color=C_HEADER, fontsize=10.5, fontfamily='monospace',
            fontweight='bold', ha='center', va='center')

# Línea separadora bajo cabecera
ax.axhline(y_top - HROW_H, color='#3a3a5a', linewidth=1.2, xmin=0, xmax=1)

# ── Filas de datos ─────────────────────────────────────────────────────────────
y_cursor = y_top - HROW_H

for idx, (ep, tl, vl, ta, va, hito, bg, c_hito) in enumerate(rows):
    row_y = y_cursor - ROW_H

    # Fondo fila
    ax.add_patch(FancyBboxPatch((0, row_y), 1, ROW_H,
        boxstyle='square,pad=0', facecolor=bg, edgecolor='none'))

    # Valores numéricos
    values = [ep, tl, vl, ta, va]
    for i, (val, xc) in enumerate(zip(values, col_centers[:-1])):
        # Colorear epoch destacado
        if idx == 6 and i == 0:   clr = C_GOLD
        elif idx == 2 and i == 0: clr = C_YELLOW
        elif idx == 7 and i == 0: clr = C_RED
        elif i in (1, 2):         clr = C_ORANGE   # losses
        elif i in (3, 4):         clr = C_GREEN    # accuracies
        else:                     clr = C_BLUE     # época
        ax.text(xc, row_y + ROW_H/2, val,
                color=clr, fontsize=10, fontfamily='monospace',
                ha='center', va='center')

    # Hito (última columna, alineado a la izquierda)
    ax.text(x_starts[-1] + 0.008, row_y + ROW_H/2, hito,
            color=c_hito, fontsize=9.5, fontfamily='monospace',
            ha='left', va='center')

    # Línea divisoria entre filas
    ax.axhline(row_y, color='#2d2d4a', linewidth=0.6, xmin=0, xmax=1)

    y_cursor = row_y

# ── Marco exterior ─────────────────────────────────────────────────────────────
for spine in ['top','bottom','left','right']:
    ax.spines[spine].set_visible(False)

rect = FancyBboxPatch((0, y_cursor), 1, y_top - y_cursor,
    boxstyle='square,pad=0', linewidth=1.2,
    edgecolor='#3a3a5a', facecolor='none',
    transform=ax.transData)
ax.add_patch(rect)

# ── Nota al pie ────────────────────────────────────────────────────────────────
note_y = y_cursor - 0.28
ax.text(0.015, note_y,
        '⬛ Fin Fase 1  |  🏆 Best checkpoint (Val Acc 86.5 %)  |  ⚠️  Early Stopping activo — gap train/val = 2.3 pts',
        color=C_GREY, fontsize=8.5, fontfamily='monospace', ha='left', va='center')

# ── Guardado ───────────────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUT_DIR, '..', '..', 'reports', 'figures')

fname = 'tabla_curva_aprendizaje.png'
out_path = os.path.join(OUT_DIR, fname)
fig_path = os.path.join(FIGURES_DIR, fname)

plt.savefig(out_path, dpi=180, bbox_inches='tight',
            facecolor=BG_OUTER, edgecolor='none')
plt.close()

shutil.copy2(out_path, fig_path)
print(f"✅  Guardado en: {out_path}")
print(f"✅  Copiado  en: {fig_path}")
