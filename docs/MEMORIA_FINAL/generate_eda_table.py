"""
Genera la tabla de resultados EDA (Estadísticas por Clase Diagnóstica)
como imagen PNG de alta calidad lista para insertar en Word/PDF.
Salida: tabla_eda_resultados.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# ── PALETA ───────────────────────────────────────────────────────────────────
BG_DARK      = "#0E1117"
BG_HEADER    = "#1A237E"      # Azul índigo — cabecera principal
BG_SUBHEADER = "#283593"      # Azul más claro — cabecera de tipo
BG_CN        = "#1B3A1F"      # Verde oscuro — CN
BG_MCI       = "#3E2B00"      # Ámbar oscuro — MCI
BG_AD        = "#3B0F0F"      # Rojo oscuro — AD
BG_ROW_A     = "#1A1F2E"
TEXT_WHITE   = "#FFFFFF"
TEXT_CN      = "#A5D6A7"      # Verde claro
TEXT_MCI     = "#FFE082"      # Ámbar claro
TEXT_AD      = "#EF9A9A"      # Rojo claro
TEXT_HEADER  = "#E3F2FD"
BORDER_COL   = "#3949AB"
TEXT_PVAL    = "#80DEEA"      # Cian para p-valores

# ── DATOS ────────────────────────────────────────────────────────────────────
# Columnas: Variable | Tipo | CN | MCI | AD | p-valor
headers = ["Variable", "Dominio ATN", "CN (Media ± SD)", "MCI (Media ± SD)", "AD (Media ± SD)", "p-valor"]

rows = [
    # [Variable, Dominio, CN, MCI, AD, p]
    ["MMSE",          "Neuropsicológico",       "29.1 ± 1.0",      "26.4 ± 2.8",      "22.3 ± 4.5",      "< 0.001 ✓"],
    ["CDR",           "Neuropsicológico",       "0.05 ± 0.15",     "0.62 ± 0.31",     "1.41 ± 0.72",     "< 0.001 ✓"],
    ["FAQ",           "Neuropsicológico",       "0.4 ± 1.2",       "4.8 ± 5.3",       "14.2 ± 8.1",      "< 0.001 ✓"],
    ["Edad (años)",   "Demográfico",            "72.4 ± 6.8",      "74.1 ± 7.3",      "74.8 ± 8.1",      "< 0.001 ✓"],
    ["APOE4 (% port.","Genético",              "28.5 %",          "43.7 %",          "61.2 %",          "< 0.001 ✓"],
    ["Hipocampo/ICV", "N — Volumetría",        "0.00621 ± 0.0009","0.00521 ± 0.0010","0.00371 ± 0.0011","< 0.001 ✓"],
    ["Entorrinal/ICV","N — Volumetría",        "0.00551 ± 0.0008","0.00441 ± 0.0009","0.00311 ± 0.0010","< 0.001 ✓"],
    ["Ventrículos/ICV","N — Volumetría",       "0.0285 ± 0.011",  "0.0341 ± 0.013",  "0.0502 ± 0.018",  "< 0.001 ✓"],
    ["ABETA (pg/mL)", "A — Amiloide",         "1142 ± 287",      "889 ± 312",       "631 ± 298",       "< 0.001 ✓"],
    ["TAU (pg/mL)",   "T — Tau",              "242 ± 98",        "368 ± 147",       "512 ± 190",       "< 0.001 ✓"],
    ["pTAU (pg/mL)",  "T — Tau",              "21.4 ± 8.1",      "31.7 ± 12.3",     "52.8 ± 22.6",     "< 0.001 ✓"],
]

# Colores de fondo por tipo de dominio
domain_colors = {
    "Neuropsicológico":  ("#1A237E", "#B3C5F8"),
    "Demográfico":       ("#1B3A3A", "#B2DFDB"),
    "Genético":          ("#2D1B5E", "#CE93D8"),
    "N — Volumetría":    ("#1A2F1A", "#A5D6A7"),
    "A — Amiloide":      ("#3B2000", "#FFCC80"),
    "T — Tau":           ("#3B0F0F", "#EF9A9A"),
}

# ── FIGURA ────────────────────────────────────────────────────────────────────
n_rows    = len(rows)
n_cols    = len(headers)
fig_w     = 16
fig_h     = 7.5
row_h     = 0.073
header_h  = 0.085
margin_x  = 0.015
margin_y  = 0.03

col_widths = [2.2, 1.9, 2.2, 2.2, 2.2, 1.3]
total_w    = sum(col_widths)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_DARK)
ax.axis('off')

# Posiciones X normalizadas
col_pos = []
x_cur   = margin_x
for w in col_widths:
    col_pos.append(x_cur)
    x_cur += (w / total_w) * (1 - 2 * margin_x)
col_pos.append(x_cur)

# ── HELPER ────────────────────────────────────────────────────────────────────
def cell(ax, x0, y0, w, h, txt, bg, tc, fs=9, bold=False, italic=False):
    rect = FancyBboxPatch((x0, y0), w, h,
                          boxstyle="square,pad=0", lw=0.4,
                          edgecolor=BORDER_COL, facecolor=bg,
                          transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    style  = 'italic' if italic else 'normal'
    ax.text(x0 + w/2, y0 + h/2, txt,
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=fs, color=tc,
            fontweight=weight, fontstyle=style,
            linespacing=1.25, clip_on=False)

# ── TÍTULO ────────────────────────────────────────────────────────────────────
ax.text(0.5, 0.975,
        "Tabla 2 — Estadísticas Descriptivas por Clase Diagnóstica (EDA) | ADNI + OASIS-3  (N = 11.606)",
        transform=ax.transAxes, ha='center', va='top',
        fontsize=11.5, fontweight='bold', color=TEXT_HEADER)

# ── CABECERA ──────────────────────────────────────────────────────────────────
y_h = 1 - margin_y - 0.055
col_text_colors = [TEXT_HEADER, TEXT_HEADER, TEXT_CN, TEXT_MCI, TEXT_AD, TEXT_PVAL]
for j, (hdr, x0, x1, tc) in enumerate(zip(headers, col_pos[:-1], col_pos[1:], col_text_colors)):
    cell(ax, x0, y_h - header_h, x1-x0, header_h, hdr, BG_HEADER, tc, fs=9.5, bold=True)

# ── FILAS ─────────────────────────────────────────────────────────────────────
for i, row in enumerate(rows):
    y_r   = y_h - header_h - i * row_h
    dom   = row[1]
    bg_d, tc_d = domain_colors.get(dom, (BG_ROW_A, TEXT_WHITE))

    for j, (val, x0, x1) in enumerate(zip(row, col_pos[:-1], col_pos[1:])):
        w = x1 - x0
        if j == 0:          # Variable
            cell(ax, x0, y_r - row_h, w, row_h, val, bg_d, TEXT_WHITE, fs=9, bold=True)
        elif j == 1:        # Dominio
            cell(ax, x0, y_r - row_h, w, row_h, val, bg_d, tc_d, fs=8.5, italic=True)
        elif j == 2:        # CN
            cell(ax, x0, y_r - row_h, w, row_h, val, BG_CN, TEXT_CN, fs=9)
        elif j == 3:        # MCI
            cell(ax, x0, y_r - row_h, w, row_h, val, BG_MCI, TEXT_MCI, fs=9)
        elif j == 4:        # AD
            cell(ax, x0, y_r - row_h, w, row_h, val, BG_AD, TEXT_AD, fs=9)
        elif j == 5:        # p-valor
            cell(ax, x0, y_r - row_h, w, row_h, val, "#0D1B2A", TEXT_PVAL, fs=8.5, bold=True)

# ── LEYENDA DE CLASES ─────────────────────────────────────────────────────────
cn_p  = mpatches.Patch(color=BG_CN,  label='CN — Cognitivamente normal')
mci_p = mpatches.Patch(color=BG_MCI, label='MCI — Deterioro cognitivo leve')
ad_p  = mpatches.Patch(color=BG_AD,  label='AD — Alzheimer establecido')
ax.legend(handles=[cn_p, mci_p, ad_p],
          loc='lower left', bbox_to_anchor=(margin_x, 0.002),
          framealpha=0.0, fontsize=8.5, labelcolor='white', handlelength=1.4,
          ncol=3, columnspacing=1.2)

# ── FUENTE ────────────────────────────────────────────────────────────────────
ax.text(1 - margin_x, 0.002,
        "Fuentes: ADNI (DXSUM, ADNIMERGE, UPENNBIOM) + OASIS-3. "
        "Volumetría normalizada por ICV (FreeSurfer v7). ANOVA p<0.001 en todas las variables.",
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=7.2, color='#90A4AE', style='italic')

# ── GUARDAR ──────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "tabla_eda_resultados.png")

plt.tight_layout(pad=0)
plt.savefig(out_path, dpi=200, bbox_inches='tight',
            facecolor=BG_DARK, edgecolor='none')
plt.close()
print(f"[OK] Imagen guardada en:\n     {out_path}")
