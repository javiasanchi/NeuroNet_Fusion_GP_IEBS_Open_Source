"""
Genera los 3 bloques de codigo Python del Capitulo 6 (EDA)
como imagenes PNG con sintaxis coloreada estilo VS Code Dark+.
Compacto y limpio, listo para Word.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, shutil

BG_EDITOR   = '#1E1E2E'
BG_GUTTER   = '#181825'
BG_TITLEBAR = '#11111B'
TEXT_DEFAULT= '#CDD6F4'
TEXT_KEYWORD= '#CBA6F7'
TEXT_FUNC   = '#89B4FA'
TEXT_STRING = '#A6E3A1'
TEXT_COMMENT= '#6C7086'
TEXT_NUMBER = '#FAB387'
TEXT_PARAM  = '#F9E2AF'
TEXT_LINE   = '#45475A'
BORDER_COL  = '#313244'

K = TEXT_KEYWORD
F = TEXT_FUNC
S = TEXT_STRING
C = TEXT_COMMENT
N = TEXT_NUMBER
P = TEXT_PARAM
D = TEXT_DEFAULT

# Cada linea = lista de (texto, color)
BLOCKS = [
    {
        'title': 'Codigo 6.1 - Estadisticas Descriptivas por Clase Diagnostica (EDA)',
        'fname': 'codigo_6_1_eda_estadisticas.png',
        'lines': [
            [('import', K), (' pandas ', D), ('as', K), (' pd', F)],
            [('import', K), (' matplotlib.pyplot ', D), ('as', K), (' plt', F)],
            [('import', K), (' seaborn ', D), ('as', K), (' sns', F)],
            [],
            [('df', D), (' = ', D), ('pd', F), ('.', D), ('read_csv', F),
             ('(', D), ("'data/ADNI_Refined_Metadata.csv'", S), (')', D)],
            [],
            [('# Estadisticas descriptivas por clase diagnostica', C)],
            [('resumen', D), (' = ', D), ('df', F), ('.', D), ('groupby', F), ("('DX')", D)],
            [('    [[', D), ("'MMSE'", S), (', ', D), ("'AGE'", S), (', ', D),
             ("'Hippocampus'", S), (', ', D), ("'Entorhinal'", S), (',', D)],
            [('      ', D), ("'TAU'", S), (', ', D), ("'ABETA'", S), (']]', D)],
            [('    .', D), ('agg', F), ("(['", D), ('mean', S), ("', '", D),
             ('std', S), ("', '", D), ('median', S), ("']).", D), ('round', F),
             ('(', D), ('3', N), (')', D)],
            [('print', F), ('(resumen)', D)],
        ]
    },
    {
        'title': 'Codigo 6.2 - Matriz de Correlacion entre Biomarcadores',
        'fname': 'codigo_6_2_correlacion.png',
        'lines': [
            [('# Matriz de correlacion entre las 10 variables mas relevantes', C)],
            [('corr_vars', D), (' = [', D)],
            [("    '", D), ('MMSE', S), ("', '", D), ('CDR', S), ("', '", D),
             ('FAQ', S), ("', '", D), ('Hippocampus', S), ("',", D)],
            [("    '", D), ('Entorhinal', S), ("', '", D), ('MidTemp', S),
             ("', '", D), ('Ventricles', S), ("',", D)],
            [("    '", D), ('TAU', S), ("', '", D), ('ABETA', S),
             ("', '", D), ('AGE', S), ("']", D)],
            [('corr_matrix', D), (' = ', D), ('df', F), ('[corr_vars].', D), ('corr', F), ('()', D)],
            [],
            [('# Correlaciones mas fuertes detectadas (ANOVA p<0.001 en todas):', C)],
            [('# MMSE  <-> Hipocampo/ICV :  r = +0.61   cognitivo <-> estructural', C)],
            [('# CDR   <-> TAU           :  r = +0.58   funcional <-> Tau', C)],
            [('# ABETA <-> Hipocampo/ICV :  r = +0.49   amiloide  <-> atrofia', C)],
        ]
    },
    {
        'title': 'Codigo 6.3 - Division Train / Test Estratificada',
        'fname': 'codigo_6_3_train_test_split.png',
        'lines': [
            [('from', K), (' sklearn.model_selection ', D), ('import', K), (' train_test_split', F)],
            [],
            [('X_train, X_test, y_train, y_test', D), (' = ', D), ('train_test_split', F), ('(', D)],
            [('    X,', D), ('            # DataFrame de 14 biomarcadores tabulares', C)],
            [('    y,', D), ('            # Etiqueta: 0=CN, 1=MCI, 2=AD', C)],
            [('    ', D), ('test_size', P), ('    = ', D), ('0.20', N), (',', D)],
            [('    ', D), ('random_state', P), (' = ', D), ('42', N), (',', D)],
            [('    ', D), ('stratify', P), ('     = y', D), ('    # Proporcion de clases identica en ambos splits', C)],
            [(')', D)],
            [],
            [('# ── Resultado del split ────────────────────────────────────────────', C)],
            [('# Train :  9.284 sujetos  (80%)  →  entrenamiento + CV-5 folds', C)],
            [('# Test  :  2.322 sujetos  (20%)  →  evaluacion final bloqueada', C)],
            [('# CN ~33%  |  MCI ~32%  |  AD ~35%   en ambos splits', C)],
        ]
    },
]

out_dir = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(out_dir, '..', '..', 'reports', 'figures')
os.makedirs(fig_dir, exist_ok=True)

FS   = 10.5   # font size codigo
LH   = 0.062  # altura de linea (normalizada)
GW   = 0.046  # ancho del gutter

for block in BLOCKS:
    lines   = block['lines']
    n       = len(lines)
    fig_h   = max(3.2, n * 0.42 + 1.4)
    fig_w   = 13.0
    pad_top = 0.09

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG_EDITOR)
    ax.set_facecolor(BG_EDITOR)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # Barra de titulo
    ax.add_patch(FancyBboxPatch(
        (0, 1 - pad_top), 1, pad_top,
        boxstyle='square,pad=0', lw=0,
        facecolor=BG_TITLEBAR,
        transform=ax.transAxes, clip_on=False))

    for xi, col in [(0.022, '#F38BA8'), (0.048, '#F9E2AF'), (0.074, '#A6E3A1')]:
        circ = plt.Circle((xi, 1 - pad_top / 2), 0.008,
                          color=col, transform=ax.transAxes, clip_on=False)
        ax.add_patch(circ)

    ax.text(0.5, 1 - pad_top / 2, block['title'],
            transform=ax.transAxes, ha='center', va='center',
            fontsize=9, color='#CDD6F4', fontweight='bold', fontfamily='monospace')

    # Gutter
    ax.add_patch(FancyBboxPatch(
        (0, 0), GW, 1 - pad_top,
        boxstyle='square,pad=0', lw=0,
        facecolor=BG_GUTTER,
        transform=ax.transAxes, clip_on=False))
    ax.plot([GW, GW], [0, 1 - pad_top],
            color=BORDER_COL, lw=0.7, transform=ax.transAxes)

    # Dibujar lineas
    content_h = 1.0 - pad_top - 0.03
    step = content_h / max(n, 1)

    for i, toks in enumerate(lines):
        y = (1 - pad_top - 0.022) - i * step - step * 0.3

        # Numero de linea
        ax.text(GW * 0.9, y, str(i + 1),
                transform=ax.transAxes, ha='right', va='center',
                fontsize=8, color=TEXT_LINE, fontfamily='monospace')

        if not toks:
            continue

        x = GW + 0.011
        for tok in toks:
            if len(tok) == 2:
                txt, col = tok
            else:
                continue
            ax.text(x, y, txt,
                    transform=ax.transAxes, ha='left', va='center',
                    fontsize=FS, color=col, fontfamily='monospace')
            x += len(txt) * 0.00745

    # Borde exterior
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle='square,pad=0', lw=1.0,
        edgecolor=BORDER_COL, facecolor='none',
        transform=ax.transAxes, clip_on=False))

    out_path = os.path.join(out_dir, block['fname'])
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor=BG_EDITOR, edgecolor='none')
    plt.close()

    shutil.copy2(out_path, fig_dir)
    print(f"[OK] {block['fname']}")

print("\nTodos los bloques generados.")
print("Ruta: E:\\MACHINE LEARNING\\proyecto_global_IEBS\\docs\\MEMORIA_FINAL\\")
