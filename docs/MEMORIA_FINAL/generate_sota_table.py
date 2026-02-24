"""
Genera la tabla comparativa SOTA del Estado del Arte (Fase 4)
como imagen PNG de alta calidad lista para insertar en Word/PDF.
Salida: tabla_sota_estado_arte.png (misma carpeta que este script)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# ── DATOS DE LA TABLA ────────────────────────────────────────────────────────
headers = ["Modelo", "Modalidad", "Dataset", "Accuracy", "F1-Score", "AUC-ROC", "Año"]

rows = [
    ["SVM + Features manuales",       "Tabular",          "ADNI",       "71%",     "0.68", "0.79", "2019"],
    ["ResNet-18 (2D cortes)",          "MRI 2D",           "ADNI",       "82%",     "0.81", "0.87", "2021"],
    ["DenseNet-121 (2D)",              "MRI 2D",           "OASIS-3",    "84%",     "0.83", "0.89", "2022"],
    ["Cross-Attention Fusion",         "MRI + Clínico",    "ADNI",       "89%",     "0.88", "0.93", "2024"],
    ["3D-ResNet50 Volumétrico",        "MRI 3D",           "ADNI",       "85%",     "0.84", "0.91", "2025"],
    ["NeuroNet-Fusion (Este trabajo)", "14 Biomarcadores\n(tabulares)",
                                                           "ADNI+OASIS", "86.5%",   "0.864","0.898","2026"],
]

# ── PALETA DE COLORES ─────────────────────────────────────────────────────────
BG_DARK    = "#0E1117"
BG_HEADER  = "#1A237E"   # Azul índigo profundo
BG_HIGHLIGHT = "#1B5E20" # Verde oscuro para nuestra fila
BG_ROW_A   = "#1A1F2E"   # Fila alternada oscura
BG_ROW_B   = "#12151F"   # Fila alternada más oscura
TEXT_WHITE  = "#FFFFFF"
TEXT_YELLOW = "#FFD54F"  # Amarillo dorado para la fila destacada
TEXT_HEADER = "#E3F2FD"  # Azul muy claro para cabecera
BORDER_COL  = "#3949AB"  # Borde azul

# ── PROPORCIONES DE COLUMNAS ──────────────────────────────────────────────────
col_widths = [3.2, 2.2, 1.5, 1.1, 1.1, 1.1, 0.9]
total_w    = sum(col_widths)
fig_width  = 14
fig_height = 5.5

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_DARK)
ax.axis('off')

# ── CÁLCULO DE COORDENADAS ────────────────────────────────────────────────────
n_rows      = len(rows)
n_cols      = len(headers)
row_height  = 0.11   # en coordenadas de figura normalizadas
header_h    = 0.13
margin_x    = 0.02
margin_y    = 0.04
x_start     = margin_x
y_start     = 1 - margin_y

# Posiciones X de columnas (normalizadas a [0,1])
col_positions = []
x_cursor = x_start
for w in col_widths:
    col_positions.append(x_cursor)
    x_cursor += (w / total_w) * (1 - 2 * margin_x)

col_positions.append(x_cursor)  # borde derecho

# ── FUNCIÓN HELPER: DIBUJAR CELDA ────────────────────────────────────────────
def draw_cell(ax, x0, y0, width, height, text, bg_color, text_color,
              fontsize=10, bold=False, valign='center', halign='center'):
    rect = FancyBboxPatch(
        (x0, y0), width, height,
        boxstyle="square,pad=0",
        linewidth=0.5,
        edgecolor=BORDER_COL,
        facecolor=bg_color,
        transform=ax.transAxes,
        clip_on=False
    )
    ax.add_patch(rect)

    weight = 'bold' if bold else 'normal'
    ax.text(
        x0 + width / 2, y0 + height / 2,
        text,
        transform=ax.transAxes,
        ha=halign, va=valign,
        fontsize=fontsize,
        color=text_color,
        fontweight=weight,
        linespacing=1.3,
        clip_on=False
    )

# ── TÍTULO DE LA TABLA ────────────────────────────────────────────────────────
ax.text(
    0.5, 0.97,
    "Tabla 1 — Comparativa SOTA: Modelos IA para Diagnóstico del Alzheimer (2019–2026)",
    transform=ax.transAxes,
    ha='center', va='top',
    fontsize=12.5, fontweight='bold',
    color=TEXT_HEADER,
    clip_on=False
)

# ── CABECERA ──────────────────────────────────────────────────────────────────
y_header = y_start - 0.08  # debajo del título

for j, (header, x0, x1) in enumerate(zip(headers,
                                          col_positions[:-1],
                                          col_positions[1:])):
    cell_w = x1 - x0
    draw_cell(ax, x0, y_header - header_h, cell_w, header_h,
              header, BG_HEADER, TEXT_HEADER,
              fontsize=10.5, bold=True)

# ── FILAS DE DATOS ────────────────────────────────────────────────────────────
for i, row in enumerate(rows):
    y_row = y_header - header_h - (i * row_height) - row_height * 0.15

    # Última fila = nuestro modelo (destacada en verde)
    is_ours = (i == len(rows) - 1)
    if is_ours:
        bg = BG_HIGHLIGHT
        tc = TEXT_YELLOW
        fs = 10.5
        bold = True
    else:
        bg = BG_ROW_A if i % 2 == 0 else BG_ROW_B
        tc = TEXT_WHITE
        fs = 10
        bold = False

    for j, (cell_text, x0, x1) in enumerate(zip(row,
                                                  col_positions[:-1],
                                                  col_positions[1:])):
        cell_w = x1 - x0
        draw_cell(ax, x0, y_row - row_height, cell_w, row_height,
                  cell_text, bg, tc,
                  fontsize=fs, bold=bold)

# ── LEYENDA ──────────────────────────────────────────────────────────────────
legend_y = y_header - header_h - (n_rows * row_height) - row_height * 0.25 - 0.04

ours_patch = mpatches.Patch(color=BG_HIGHLIGHT, label='Este trabajo — NeuroNet-Fusion')
ref_patch  = mpatches.Patch(color=BG_ROW_A,     label='Modelos de referencia (literatura)')
ax.legend(
    handles=[ours_patch, ref_patch],
    loc='lower left',
    bbox_to_anchor=(margin_x, 0.01),
    framealpha=0.0,
    fontsize=9,
    labelcolor='white',
    handlelength=1.5
)

# ── FUENTE ────────────────────────────────────────────────────────────────────
ax.text(
    1 - margin_x, 0.01,
    "Fuentes: Sánchez-García (2019), Zhang (2021), Chen (2024), Luo (2025). NeuroNet-Fusion: resultados propios.",
    transform=ax.transAxes,
    ha='right', va='bottom',
    fontsize=7.5, color='#90A4AE',
    style='italic', clip_on=False
)

# ── GUARDAR ──────────────────────────────────────────────────────────────────
out_dir  = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "tabla_sota_estado_arte.png")

plt.tight_layout(pad=0)
plt.savefig(out_path, dpi=200, bbox_inches='tight',
            facecolor=BG_DARK, edgecolor='none')
plt.close()

print(f"[OK] Imagen guardada en:\n     {out_path}")
