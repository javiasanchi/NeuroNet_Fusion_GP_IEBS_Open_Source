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
    # Aumentamos el factor de altura de 0.25 a 0.45 para máximo 'aire'
    fig_height = len(code_lines) * 0.45 + 1.5 
    fig, ax = plt.subplots(figsize=(12, fig_height))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Barra de título tipo macOS
    header_rect = patches.Rectangle((0, 0.96), 1, 0.04, transform=ax.transAxes, color=HEADER_COLOR, zorder=10)
    ax.add_patch(header_rect)
    
    # Botones macOS
    for i, col in enumerate(['#ff5f57', '#ffbd2e', '#28c840']):
        circle = plt.Circle((0.02 + i*0.02, 0.98), 0.006, transform=ax.transAxes, color=col, zorder=11)
        ax.add_artist(circle)
    
    # Título en la barra
    ax.text(0.5, 0.98, title, transform=ax.transAxes, color='#a6adc8', 
            fontsize=11, fontweight='bold', ha='center', va='center', family='monospace')

    # Contenido del código
    y_pos = 0.92
    for i, line in enumerate(code_lines):
        # Número de línea
        ax.text(0.01, y_pos, f"{i+1:2}", transform=ax.transAxes, color='#45475a', 
                fontsize=11, family='monospace', ha='right', va='center')
        
        # Procesar resaltado básico por línea
        x_offset = 0.03
        words = line.split(' ')
        for word in words:
            color = TEXT_COLOR
            # Lógica simple de resaltado
            stripped = word.strip('():,[]"\'')
            if stripped in ['class', 'def', 'import', 'from', 'as', 'return', 'super', 'if', 'else', 'for', 'in', 'with']:
                color = KEYWORD_COLOR
            elif stripped in ['torch', 'nn', 'models', 'Sequential', 'Linear', 'ReLU', 'Dropout', 'LayerNorm', 'Conv3d', 'BatchNorm3d', 'MaxPool3d', 'AdaptiveAvgPool3d', 'NeuroNetFusion', 'NeuroNet3D', 'ResNet3D_Block']:
                color = CLASS_COLOR
            elif word.startswith('#'):
                color = COMMENT_COLOR
            elif (word.endswith('(') or '(' in word) and not word.startswith('self.'):
                color = FUNCTION_COLOR
            elif '"' in word or "'" in word:
                color = STRING_COLOR
            elif stripped.isdigit():
                color = NUMBER_COLOR
            
            txt = ax.text(x_offset, y_pos, word + ' ', transform=ax.transAxes, color=color, 
                    fontsize=12, family='monospace', ha='left', va='center')
            
            # Ajustamos el factor de ancho a 0.013 para evitar el hacinamiento lateral
            x_offset += len(word) * 0.013 + 0.01
            
        # Aumentamos el salto de línea a 0.04 para evitar el solapamiento vertical
        y_pos -= 0.04

    ax.set_axis_off()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()

# Definición de bloques para Capítulo 10
BLOCKS = [
    {
        'title': '10.2.1 Arquitectura NeuroNet-Fusion (Dual Backbone CNN)',
        'fname': 'codigo_10_2_neuronet_fusion.png',
        'lines': [
            'class NeuroNetFusion(nn.Module):',
            '    """Arquitectura de Fusión Dual para clasificación de Alzheimer."""',
            '    def __init__(self, num_classes=4, pretrained=True):',
            '        super(NeuroNetFusion, self).__init__()',
            '        # BACKBONE 1: ResNet50',
            '        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)',
            '        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])',
            '        # BACKBONE 2: DenseNet121',
            '        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)',
            '        self.densenet_features = densenet.features',
            '        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))',
            '        # CLASIFICADOR DE FUSIÓN',
            '        self.classifier = nn.Sequential(',
            '            nn.Linear(2048 + 1024, 512),',
            '            nn.LayerNorm(512),',
            '            nn.ReLU(),',
            '            nn.Dropout(0.5),',
            '            nn.Linear(512, 128),',
            '            nn.LayerNorm(128),',
            '            nn.ReLU(),',
            '            nn.Linear(128, num_classes)',
            '        )',
            '    ',
            '    def forward(self, x):',
            '        f1 = torch.flatten(self.resnet_features(x), 1)',
            '        f2 = torch.flatten(self.avgpool(self.densenet_features(x)), 1)',
            '        fused = torch.cat((f1, f2), dim=1)',
            '        return self.classifier(fused)'
        ]
    },
    {
        'title': '10.3 Arquitectura NeuroNet3D (ResNet3D 128x128x128)',
        'fname': 'codigo_10_3_neuronet3d.png',
        'lines': [
            'class NeuroNet3D(nn.Module):',
            '    """3D-ResNet para clasificación de volúmenes MRI."""',
            '    def __init__(self, num_classes=3):',
            '        super().__init__()',
            '        self.stem = nn.Sequential(',
            '            nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3),',
            '            nn.BatchNorm3d(32), nn.ReLU(),',
            '            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)',
            '        )',
            '        self.layer1 = ResNet3D_Block(32, 64)',
            '        self.layer2 = ResNet3D_Block(64, 128, stride=2)',
            '        self.layer3 = ResNet3D_Block(128, 256, stride=2)',
            '        self.pool = nn.AdaptiveAvgPool3d(1)',
            '        self.head = nn.Linear(256, num_classes)',
            '    ',
            '    def forward(self, x):',
            '        x = self.stem(x)',
            '        x = self.layer3(self.layer2(self.layer1(x)))',
            '        return self.head(torch.flatten(self.pool(x), 1))'
        ]
    }
]

# Directorios
OUT_DIR = r'E:\MACHINE LEARNING\proyecto_global_IEBS\docs\MEMORIA_FINAL'
FIGURES_DIR = r'E:\MACHINE LEARNING\proyecto_global_IEBS\reports\figures'

# Generar imágenes
for block in BLOCKS:
    path = os.path.join(OUT_DIR, block['fname'])
    render_code_snippet(block['title'], block['lines'], path)
    # Copiar a figures
    shutil.copy2(path, os.path.join(FIGURES_DIR, block['fname']))
    print(f"Generada: {path}")
