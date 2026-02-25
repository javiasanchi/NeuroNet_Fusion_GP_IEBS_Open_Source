import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import shutil

# Configuración de colores (VS Code Dark+ Pro)
BG_COLOR = '#1e1e2e'
HEADER_COLOR = '#181825'
TEXT_COLOR = '#cdd6f4'
KEYWORD_COLOR = '#c678dd'  # Púrpura
STRING_COLOR = '#98c379'   # Verde
COMMENT_COLOR = '#676e95'  # Gris azulado
FUNCTION_COLOR = '#61afef' # Azul
CLASS_COLOR = '#e5c07b'    # Amarillo
DECORATOR_COLOR = '#56b6c2' # Cian
NUMBER_COLOR = '#d19a66'   # Naranja

def render_code_snippet(title, code_lines, filename):
    fig_height = len(code_lines) * 0.4 + 1.5 
    fig, ax = plt.subplots(figsize=(10, fig_height))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Barra de título tipo macOS
    header_rect = patches.Rectangle((0, 0.95), 1, 0.05, transform=ax.transAxes, color=HEADER_COLOR, zorder=10)
    ax.add_patch(header_rect)
    
    # Botones macOS
    for i, col in enumerate(['#ff5f57', '#ffbd2e', '#28c840']):
        circle = plt.Circle((0.02 + i*0.02, 0.975), 0.006, transform=ax.transAxes, color=col, zorder=11)
        ax.add_artist(circle)
    
    # Título en la barra
    ax.text(0.5, 0.975, title, transform=ax.transAxes, color='#a6adc8', 
            fontsize=10, fontweight='bold', ha='center', va='center', family='monospace')

    # Contenido del código
    y_pos = 0.92
    for i, line in enumerate(code_lines):
        ax.text(0.01, y_pos, f"{i+1:2}", transform=ax.transAxes, color='#45475a', 
                fontsize=11, family='monospace', ha='right', va='center')
        
        x_offset = 0.03
        words = line.split(' ')
        for word in words:
            color = TEXT_COLOR
            stripped = word.strip('():,[]"\'')
            if stripped in ['class', 'def', 'import', 'from', 'as', 'return', 'super', 'if', 'else', 'for', 'in', 'with']:
                color = KEYWORD_COLOR
            elif word.startswith('#'):
                color = COMMENT_COLOR
            elif word.endswith(':') and not (word.startswith("'") or word.startswith('"')):
                color = FUNCTION_COLOR
            elif '"' in word or "'" in word:
                color = STRING_COLOR
            elif stripped.replace('.','').isdigit():
                color = NUMBER_COLOR
            
            txt = ax.text(x_offset, y_pos, word + ' ', transform=ax.transAxes, color=color, 
                    fontsize=12, family='monospace', ha='left', va='center')
            
            x_offset += len(word) * 0.013 + 0.01
            
        y_pos -= 0.035

    ax.set_axis_off()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()

# Bloques Capítulo 9.4
BLOCKS = [
    {
        'title': '9.4 Hiperparámetros Óptimos XGBoost (Best Optuna Trial)',
        'fname': 'codigo_9_4_best_params.png',
        'lines': [
            'best_params = {',
            "    'n_estimators':     850,",
            "    'max_depth':        6,",
            "    'learning_rate':    0.048,",
            "    'subsample':        0.82,",
            "    'colsample_bytree': 0.75,",
            "    'reg_alpha':        0.12,   # L1 Lasso",
            "    'reg_lambda':       1.45,   # L2 Ridge",
            "    'gamma':            0.18,",
            "    'tree_method':      'hist', # GPU Hist",
            "    'device':           'cuda', # RTX 4070",
            '}'
        ]
    }
]

OUT_DIR = r'E:\MACHINE LEARNING\proyecto_global_IEBS\docs\MEMORIA_FINAL'
FIGURES_DIR = r'E:\MACHINE LEARNING\proyecto_global_IEBS\reports\figures'

for block in BLOCKS:
    path = os.path.join(OUT_DIR, block['fname'])
    render_code_snippet(block['title'], block['lines'], path)
    shutil.copy2(path, os.path.join(FIGURES_DIR, block['fname']))
    print(f"Generada: {path}")
