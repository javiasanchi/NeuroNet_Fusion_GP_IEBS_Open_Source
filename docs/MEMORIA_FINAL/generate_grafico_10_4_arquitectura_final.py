"""
Genera el diagrama de Arquitectura de Producción (10.4)
como imagen PNG — estilo VS Code Dark+.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow
import os, shutil

# ── Paleta VS Code Dark+ ──────────────────────────────────────────────────────
BG_OUTER   = '#1e1e2e'
BG_INNER   = '#1e1f2b'
BG_BOX     = '#252640'

C_WHITE    = '#d4d4d4'
C_BLUE     = '#569cd6'
C_PURPLE   = '#c586c0'
C_ORANGE   = '#ce9178'
C_GREEN    = '#4ec9b0'
C_YELLOW   = '#dcdcaa'
C_GREY     = '#6a737d'

# ── Configuración ─────────────────────────────────────────────────────────────
FIG_W = 12
FIG_H = 9

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG_OUTER)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(BG_OUTER)
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.axis('off')

def draw_box(x, y, w, h, title, content, color_title=C_BLUE):
    # Sombra
    ax.add_patch(FancyBboxPatch((x+0.5, y-0.5), w, h, boxstyle='round,pad=0.2', 
                 facecolor='#000000', alpha=0.3, edgecolor='none'))
    # Box principal
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2', 
                 facecolor=BG_BOX, edgecolor='#3a3a5a', linewidth=1.5))
    
    # Título
    ax.text(x + w/2, y + h - 1.5, title, color=color_title, fontsize=12, 
            fontweight='bold', fontfamily='monospace', ha='center')
    
    # Línea separadora
    ax.plot([x + 1, x + w - 1], [y + h - 2.5, y + h - 2.5], color='#3a3a5a', lw=1)
    
    # Contenido
    y_text = y + h - 4.5
    for line in content:
        ax.text(x + 2, y_text, line, color=C_WHITE, fontsize=10.5, 
                fontfamily='monospace', ha='left')
        y_text -= 1.8

def draw_arrow(x, y, length, direction='down'):
    if direction == 'down':
        ax.annotate('', xy=(x, y-length), xytext=(x, y),
                    arrowprops=dict(facecolor=C_GREY, edgecolor=C_GREY, width=2, headwidth=10))

# ── 1. INPUT ──────────────────────────────────────────────────────────────────
draw_box(30, 82, 40, 12, " [ ENTRADA DE DATOS ] ", [
    "• 14 Biomarcadores Clínicos",
    "• Datos Tabulares (ATN Frame)"
], color_title=C_YELLOW)

draw_arrow(50, 81, 6)

# ── 2. XGBOOST CHAMPION ───────────────────────────────────────────────────────
draw_box(30, 58, 40, 16, " [ XGBOOST CHAMPION ] ", [
    "• n_estimators: 850",
    "• max_depth: 6",
    "• learning_rate: 0.048",
    "• AUC ROC: 0.898 (Validado)"
], color_title=C_GREEN)

draw_arrow(50, 57, 6)

# ── 3. ENSEMBLE SOFT VOTING ───────────────────────────────────────────────────
draw_box(25, 33, 50, 17, " [ ENSEMBLE: VOTACIÓN BLANDA ] ", [
    "• XGBoost (Peso 0.5) — Estabilidad",
    "• LightGBM (Peso 0.3) — Velocidad",
    "• CatBoost (Peso 0.2) — Robustez Outliers",
    "• Reducción Varianza: -42%"
], color_title=C_PURPLE)

draw_arrow(50, 32, 6)

# ── 4. OUTPUT ─────────────────────────────────────────────────────────────────
draw_box(30, 8, 40, 17, " [ SALIDA Y DICTAMEN ] ", [
    "• Probabilidades: P(CN), P(MCI), P(AD)",
    "• Clasificación Perfil ATN",
    "• Dictamen Clínico Narrativo",
    "• Grado de Incertidumbre (Softmax)"
], color_title=C_BLUE)

# ── Título Superior ───────────────────────────────────────────────────────────
ax.text(50, 97, "FIGURA 10.4: FLUJO DE LA ARQUITECTURA DE PRODUCCIÓN (NEURONET-FUSION)", 
        color=C_WHITE, fontsize=13, fontweight='bold', fontfamily='monospace', ha='center')

# ── Guardado ───────────────────────────────────────────────────────────────────
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUT_DIR, '..', '..', 'reports', 'figures')

fname = 'grafico_10_4_arquitectura_final.png'
out_path = os.path.join(OUT_DIR, fname)
fig_path = os.path.join(FIGURES_DIR, fname)

plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG_OUTER)
plt.close()

shutil.copy2(out_path, fig_path)
print(f"✅  Generado y copiado: {fname}")
