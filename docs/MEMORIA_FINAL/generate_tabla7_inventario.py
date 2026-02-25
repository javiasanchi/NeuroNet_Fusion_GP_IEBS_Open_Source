"""
Genera la Tabla 7.1 — Inventario de Variables de Entrada (14 biomarcadores)
como imagen PNG de alta calidad, con colores por dominio ATN.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os, shutil

out_dir = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(out_dir, '..', '..', 'reports', 'figures')
os.makedirs(fig_dir, exist_ok=True)

# ── PALETA GENERAL ────────────────────────────────────────────────────────────
BG_DARK     = '#0F1117'
BG_HEADER   = '#1C1F2E'
BG_TITLE    = '#111827'
TEXT_MAIN   = '#F8FAFC'
TEXT_HEAD   = '#93C5FD'
TEXT_MUTED  = '#94A3B8'
TEXT_SUB    = '#CBD5E1'
BORDER      = '#334155'

# Colores por dominio ATN
DOMAIN_COLORS = {
    'Neuropsicológico': {'bg': '#1E2D3D', 'pill': '#3B82F6', 'text': '#93C5FD'},
    'Demográfico':      {'bg': '#1E2A1E', 'pill': '#22C55E', 'text': '#86EFAC'},
    'Genético':         {'bg': '#2D1E3D', 'pill': '#A855F7', 'text': '#D8B4FE'},
    'N — Volumetría':   {'bg': '#2D2210', 'pill': '#F59E0B', 'text': '#FCD34D'},
    'A — Amiloide':     {'bg': '#2D1E1E', 'pill': '#EF4444', 'text': '#FCA5A5'},
    'T — Tau':          {'bg': '#1E2D2D', 'pill': '#06B6D4', 'text': '#67E8F9'},
}

# ── DATOS DE LA TABLA ─────────────────────────────────────────────────────────
# Columnas: Variable | Tipo | Fuente | Rango esperado | Dominio
rows = [
    # Variable             Tipo         Fuente           Rango                   Dominio
    ('MMSE',               'Continua',  'ADNI / OASIS',  '0 – 30 puntos',        'Neuropsicológico'),
    ('CDR',                'Ordinal',   'ADNI',           '0 / 0.5 / 1 / 2 / 3', 'Neuropsicológico'),
    ('FAQ',                'Continua',  'ADNI',           '0 – 30 puntos',        'Neuropsicológico'),
    ('ADAS-11',            'Continua',  'ADNI',           '0 – 70 puntos',        'Neuropsicológico'),
    ('AGE',                'Continua',  'Demografía',     '50 – 90 años',         'Demográfico'),
    ('EDUCYEARS',          'Continua',  'Demografía',     '6 – 20 años',          'Demográfico'),
    ('APOE4',              'Binaria',   'Genética',       '0 (No) / 1 (Portador)','Genético'),
    ('Hippocampus / ICV',  'Continua',  'FreeSurfer',     '0.001 – 0.010',        'N — Volumetría'),
    ('Entorhinal / ICV',   'Continua',  'FreeSurfer',     '0.001 – 0.010',        'N — Volumetría'),
    ('MidTemporal / ICV',  'Continua',  'FreeSurfer',     '0.005 – 0.020',        'N — Volumetría'),
    ('Ventricles / ICV',   'Continua',  'FreeSurfer',     '0.01 – 0.15',          'N — Volumetría'),
    ('ABETA-42',           'Continua',  'LCR (ADNI)',     '200 – 2000 pg/mL',     'A — Amiloide'),
    ('TAU Total',          'Continua',  'LCR (ADNI)',     '50 – 1200 pg/mL',      'T — Tau'),
    ('pTau-181',           'Continua',  'LCR (ADNI)',     '10 – 300 pg/mL',       'T — Tau'),
]

COLS      = ['Variable', 'Tipo', 'Fuente', 'Rango Esperado', 'Dominio ATN']
N_ROWS    = len(rows)
N_COLS    = len(COLS)

# Anchos relativos de columna
COL_W = [0.18, 0.10, 0.14, 0.22, 0.36]   # suma = 1.0

# ── FIGURA ────────────────────────────────────────────────────────────────────
fig_w, fig_h = 16, 9.2
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_DARK)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# ── TÍTULO ────────────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.01, 0.91), 0.98, 0.085,
    boxstyle='round,pad=0.01', lw=0, facecolor=BG_TITLE,
    transform=ax.transAxes, clip_on=False))

ax.text(0.5, 0.952, 'Tabla 7.1 — Inventario de Variables de Entrada del Modelo de Producción',
        transform=ax.transAxes, ha='center', va='center',
        fontsize=14, fontweight='bold', color=TEXT_MAIN,
        fontfamily='sans-serif')

ax.text(0.5, 0.918, '14 biomarcadores clínicos y moleculares · Modelo XGBoost · No incluye imágenes MRI directas',
        transform=ax.transAxes, ha='center', va='center',
        fontsize=9.5, color=TEXT_MUTED, fontstyle='italic')

# ── TABLA ─────────────────────────────────────────────────────────────────────
table_top  = 0.895
table_bot  = 0.055
table_h    = table_top - table_bot
row_h      = table_h / (N_ROWS + 1)   # +1 para header
table_left = 0.01
table_right= 0.99
table_w    = table_right - table_left

# Acumular posiciones X de columnas
col_x = [table_left]
for w in COL_W[:-1]:
    col_x.append(col_x[-1] + w * table_w)

def cell_x_center(ci):
    return col_x[ci] + COL_W[ci] * table_w / 2

# ── ENCABEZADO ────────────────────────────────────────────────────────────────
hdr_y = table_top - row_h
ax.add_patch(FancyBboxPatch(
    (table_left, hdr_y), table_w, row_h,
    boxstyle='square,pad=0', lw=0, facecolor=BG_HEADER,
    transform=ax.transAxes, clip_on=False))

for ci, col in enumerate(COLS):
    ax.text(cell_x_center(ci), hdr_y + row_h * 0.5, col,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=TEXT_HEAD)

# Línea inferior del header
ax.plot([table_left, table_right], [hdr_y, hdr_y],
        color=TEXT_HEAD, lw=1.2, transform=ax.transAxes, alpha=0.7)

# ── FILAS DE DATOS ────────────────────────────────────────────────────────────
PILL_H  = 0.026   # altura del pill del dominio (ax units)
PILL_R  = 0.008   # radio del pill

for ri, row in enumerate(rows):
    var, tipo, fuente, rango, dominio = row
    dom  = DOMAIN_COLORS[dominio]
    ry   = hdr_y - (ri + 1) * row_h
    even = ri % 2 == 0

    # Fondo fila (alternado sutil)
    row_bg = '#161B2E' if even else '#111520'
    ax.add_patch(FancyBboxPatch(
        (table_left, ry), table_w, row_h,
        boxstyle='square,pad=0', lw=0, facecolor=row_bg,
        transform=ax.transAxes, clip_on=False))

    cy = ry + row_h * 0.5   # centro vertical de fila

    # Col 0 — Variable (negrita, blanco)
    ax.text(cell_x_center(0), cy, var,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=9.8, fontweight='bold', color=TEXT_MAIN)

    # Col 1 — Tipo
    ax.text(cell_x_center(1), cy, tipo,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=9.2, color=TEXT_SUB)

    # Col 2 — Fuente
    ax.text(cell_x_center(2), cy, fuente,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=9.2, color=TEXT_SUB)

    # Col 3 — Rango (en monospace)
    ax.text(cell_x_center(3), cy, rango,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=9.2, color='#E2E8F0', fontfamily='monospace')

    # Col 4 — Dominio ATN (pill coloreado)
    px = col_x[4] + COL_W[4] * table_w * 0.5
    pill_w_ax = COL_W[4] * table_w * 0.72   # ancho del pill en ax units
    pill_h_ax = row_h * 0.55

    # Ajustar fig-units reales
    ax_bbox = ax.get_position()
    pw_fig  = pill_w_ax * ax_bbox.width
    ph_fig  = pill_h_ax * ax_bbox.height

    # FancyBboxPatch en ax-coords
    ax.add_patch(FancyBboxPatch(
        (px - pill_w_ax / 2, cy - pill_h_ax / 2),
        pill_w_ax, pill_h_ax,
        boxstyle=f'round,pad=0.008',
        lw=1.2, edgecolor=dom['pill'],
        facecolor=dom['bg'],
        transform=ax.transAxes, clip_on=True))

    # Punto de color + texto dominio
    dot_x = px - pill_w_ax / 2 + 0.018
    ax.plot(dot_x, cy, 'o', ms=5, color=dom['pill'],
            transform=ax.transAxes, zorder=5)

    ax.text(dot_x + 0.015, cy, dominio,
            transform=ax.transAxes, ha='left', va='center',
            fontsize=9.0, color=dom['text'], fontweight='bold')

    # Separador horizontal
    ax.plot([table_left, table_right], [ry, ry],
            color=BORDER, lw=0.4, transform=ax.transAxes, alpha=0.6)

# Separadores verticales columnas
for ci in range(1, N_COLS):
    ax.plot([col_x[ci], col_x[ci]], [table_bot, table_top],
            color=BORDER, lw=0.5, transform=ax.transAxes, alpha=0.5)

# Borde exterior tabla
ax.add_patch(FancyBboxPatch(
    (table_left, table_bot), table_w, table_top - table_bot,
    boxstyle='round,pad=0.005', lw=1.5,
    edgecolor='#475569', facecolor='none',
    transform=ax.transAxes, clip_on=False))

# ── PIE DE TABLA ──────────────────────────────────────────────────────────────
ax.text(0.5, 0.022,
        'Fuente: ADNI (alzheimer.loni.usc.edu) · OASIS-3 (oasis-brains.org) · '
        'Volumetría derivada de segmentación FreeSurfer · LCR = Líquido Cefalorraquídeo · '
        'ICV = Intracranial Volume (normalización)',
        transform=ax.transAxes, ha='center', va='center',
        fontsize=7.8, color=TEXT_MUTED, fontstyle='italic')

# ── LEYENDA DOMINIOS ──────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=v['bg'], edgecolor=v['pill'], label=k, linewidth=1.5)
    for k, v in DOMAIN_COLORS.items()
]
leg = ax.legend(handles=legend_patches, loc='lower center',
                bbox_to_anchor=(0.5, -0.005),
                ncol=6, fontsize=8, framealpha=0.0,
                labelcolor=[v['text'] for v in DOMAIN_COLORS.values()],
                handlelength=1.2, handleheight=0.8,
                borderpad=0.3, columnspacing=1.0)
for t in leg.get_texts():
    t.set_fontweight('bold')

plt.tight_layout(pad=0.3)
out_path = os.path.join(out_dir, 'tabla_7_inventario_variables.png')
plt.savefig(out_path, dpi=180, bbox_inches='tight',
            facecolor=BG_DARK, edgecolor='none')
plt.close()
shutil.copy2(out_path, fig_dir)
print(f"[OK] tabla_7_inventario_variables.png")
print(f"[COPY] → reports/figures/")
