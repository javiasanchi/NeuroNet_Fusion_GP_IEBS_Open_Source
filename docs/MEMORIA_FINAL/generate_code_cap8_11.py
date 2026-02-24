"""
Genera los bloques de código Python del Capítulo 8 (Ingeniería de Características)
y Capítulo 11 (Entrenamiento y Optimización) como imágenes PNG individuales.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, shutil

out_dir = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(out_dir, '..', '..', 'reports', 'figures')
os.makedirs(fig_dir, exist_ok=True)

# ── PALETA ─────────────────────────────────────────────────────────────────────
BG  = '#1E1E2E'; GUT = '#181825'; BAR = '#11111B'
D   = '#CDD6F4'; K   = '#CBA6F7'; F   = '#89B4FA'
S   = '#A6E3A1'; C   = '#6C7086'; N   = '#FAB387'
P   = '#F9E2AF'; T   = '#F38BA8'; LN  = '#45475A'
BO  = '#313244'; TP  = '#F38BA8'; TY  = '#F9E2AF'; TG  = '#A6E3A1'

# ── BLOQUES ────────────────────────────────────────────────────────────────────
BLOCKS = [

    # ── 8.1 ── FeatureExtractor ResNet50 ──────────────────────────────────────
    {
        'title': '8.1  Features de Imagen — Deep Embeddings con ResNet50 (2048-D)',
        'fname': 'codigo_8_1_feature_extractor_resnet.png',
        'lines': [
            [('import', K), (' torch', F)],
            [('import', K), (' torch.nn ', D), ('as', K), (' nn', F)],
            [('from', K), (' torchvision ', D), ('import', K), (' models', F), (', ', D), ('transforms', F)],
            [('from', K), (' PIL ', D), ('import', K), (' Image', F)],
            [],
            [('class', K), (' FeatureExtractor', T), ('(', D), ('nn.Module', F), ('):', D)],
            [('    """Extrae el vector de embedding de 2048 dimensiones de ResNet50."""', C)],
            [('    def', K), (' __init__', F), ('(self):', D)],
            [('        super', F), ('().__init__()', D)],
            [('        resnet = ', D), ('models', F), ('.resnet50(weights=', D), ('models', F),
             ('.ResNet50_Weights.DEFAULT)', D)],
            [('        # Eliminar la capa de clasificacion final', C)],
            [('        self.backbone = ', D), ('nn.Sequential', T),
             ('(*list(resnet.children())[:-', D), ('1', N), ('])', D)],
            [],
            [('    def', K), (' forward', F), ('(self, x):', D)],
            [('        return', K), (' torch', F), ('.flatten(self.backbone(x), ', D), ('1', N),
             (')  ', D), ('# (B, 2048)', C)],
            [],
            [('extractor = ', D), ('FeatureExtractor', T), ('().eval().cuda()', D)],
            [],
            [('# Extraccion de embeddings para un batch de MRIs', C)],
            [('def', K), (' extract_embeddings', F),
             ('(image_paths, batch_size=', D), ('32', N), ('):', D)],
            [('    transform = ', D), ('transforms', F), ('.Compose([', D)],
            [('        transforms', F), ('.Resize((', D), ('224', N), (', ', D), ('224', N), (')),', D)],
            [('        transforms', F), ('.ToTensor(),', D)],
            [('        transforms', F), ('.Normalize([', D),
             ('0.485', N), (', ', D), ('0.456', N), (', ', D), ('0.406', N),
             ('], [', D), ('0.229', N), (', ', D), ('0.224', N), (', ', D), ('0.225', N), ('])', D)],
            [('    ])', D)],
            [('    embeddings = []', D)],
            [('    for', K), (' path ', D), ('in', K), (' image_paths:', D)],
            [('        img = transform(', D), ('Image', F),
             ('.open(path).convert(', D), ("'RGB'", S), (')).unsqueeze(', D), ('0', N), (').cuda()', D)],
            [('        with', K), (' torch', F), ('.no_grad():', D)],
            [('            emb = extractor(img).cpu().numpy()', D)],
            [('        embeddings.append(emb)', D)],
            [('    return', K), (' np', F), ('.vstack(embeddings)  ', D), ('# (N, 2048)', C)],
        ]
    },

    # ── 8.2 ── ClinicalMLP 128-D ──────────────────────────────────────────────
    {
        'title': '8.2  Features Clínicos — Embedding MLP de 128 Dimensiones',
        'fname': 'codigo_8_2_clinical_mlp.png',
        'lines': [
            [('class', K), (' ClinicalMLP', T), ('(', D), ('nn.Module', F), ('):', D)],
            [('    """', C)],
            [('    Red de embedding para biomarcadores clinicos tabulares.', C)],
            [('    Input:  14 biomarcadores estandarizados', C)],
            [('    Output: Vector latente de 128 dimensiones', C)],
            [('    """', C)],
            [('    def', K), (' __init__', F),
             ('(self, input_dim=', D), ('14', N), (', latent_dim=', D), ('128', N), ('):', D)],
            [('        super', F), ('().__init__()', D)],
            [('        self.encoder = ', D), ('nn.Sequential', T), ('(', D)],
            [('            nn.Linear(input_dim, ', D), ('256', N), ('),', D)],
            [('            nn.LayerNorm(', D), ('256', N), ('),', D)],
            [('            nn.ReLU(),', D)],
            [('            nn.Dropout(', D), ('0.3', N), ('),', D)],
            [('            nn.Linear(', D), ('256', N), (', latent_dim),', D)],
            [('            nn.LayerNorm(latent_dim),', D)],
            [('            nn.ReLU()', D)],
            [('        )', D)],
            [],
            [('    def', K), (' forward', F), ('(self, x):', D)],
            [('        return', K), (' self.encoder(x)  ', D), ('# (B, 128)', C)],
        ]
    },

    # ── 8.3 ── Fusión de Características ─────────────────────────────────────
    {
        'title': '8.3  Estrategia de Fusión de Características — Concatenación + Bottleneck 2176→512',
        'fname': 'codigo_8_3_fusion_caracteristicas.png',
        'lines': [
            [('# Concatenacion de embeddings visuales y clinicos', C)],
            [('# Ejemplo con batch de 32 pacientes:', C)],
            [('F_visual   = extract_embeddings(mri_paths)[:', D), ('32', N),
             (']   ', D), ('# (32, 2048)', C)],
            [('F_clinical = clinical_mlp(X_clinical_tensor)       ', D), ('# (32, 128)', C)],
            [],
            [('F_fused = torch.cat([F_visual, F_clinical], dim=', D), ('1', N),
             (') ', D), ('# (32, 2176)', C)],
            [],
            [('# Cuello de botella (Bottleneck): 2176 → 512', C)],
            [('bottleneck = nn.Sequential(', D)],
            [('    nn.Linear(', D), ('2176', N), (', ', D), ('512', N), ('),', D)],
            [('    nn.LayerNorm(', D), ('512', N), ('),', D)],
            [('    nn.ReLU(),', D)],
            [('    nn.Dropout(', D), ('0.4', N), (')', D)],
            [(')', D)],
            [('F_final = bottleneck(F_fused)  ', D), ('# (32, 512)', C)],
            [],
            [('# Justificacion de LayerNorm en el bottleneck:', C)],
            [('#   Embeddings visuales : rango ~0-5  (activaciones ResNet50)', C)],
            [('#   Embeddings clinicos : rango ~0-1  (tras StandardScaler)', C)],
            [('#   LayerNorm equilibra magnitudes antes de la fusion', C)],
        ]
    },

    # ── 8.4 ── GLCM Features ──────────────────────────────────────────────────
    {
        'title': '8.4  Features de Textura Manual — Descriptores GLCM (Baseline Explainable)',
        'fname': 'codigo_8_4_glcm_features.png',
        'lines': [
            [('from', K), (' skimage.feature ', D), ('import', K),
             (' graycomatrix', F), (', ', D), ('graycoprops', F)],
            [('import', K), (' numpy ', D), ('as', K), (' np', F)],
            [],
            [('def', K), (' extract_glcm_features', F),
             ('(image_array: ', D), ('np.ndarray', T), (') -> ', D), ('dict', T), (':', D)],
            [('    """', C)],
            [('    Extrae caracteristicas de textura GLCM de un corte MRI axial.', C)],
            [('    Metricas: contrast, dissimilarity, homogeneity, energy, correlation, ASM', C)],
            [('    """', C)],
            [('    # Cuantizacion a 256 niveles de gris', C)],
            [('    img_uint8 = (image_array * ', D), ('255', N),
             (').astype(np.uint8)', D)],
            [],
            [('    glcm = graycomatrix(img_uint8,', D)],
            [('                distances=[', D), ('1', N), (', ', D), ('3', N), ('],', D)],
            [('                angles=[', D), ('0', N), (', np.pi/', D), ('4', N), ('],', D)],
            [('                levels=', D), ('256', N),
             (', symmetric=', D), ('True', K),
             (', normed=', D), ('True', K), (')', D)],
            [],
            [('    return', K), (' {', D)],
            [("        'contrast'     : graycoprops(glcm, 'contrast').mean(),", D)],
            [("        'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),", D)],
            [("        'homogeneity'  : graycoprops(glcm, 'homogeneity').mean(),", D)],
            [("        'energy'       : graycoprops(glcm, 'energy').mean(),", D)],
            [("        'correlation'  : graycoprops(glcm, 'correlation').mean(),", D)],
            [("        'ASM'          : graycoprops(glcm, 'ASM').mean()", D)],
            [('    }', D)],
        ]
    },

    # ── 8.5 ── Feature Columns Producción ─────────────────────────────────────
    {
        'title': '8.5  Selección Final de Features — Modelo de Producción XGBoost (14 Biomarcadores)',
        'fname': 'codigo_8_5_feature_columns.png',
        'lines': [
            [('FEATURE_COLUMNS = [', D)],
            [('    # Cognitivo (alta importancia — SHAP top-4)', C)],
            [("    'MMSE'", S), (", 'CDR'", S), (", 'FAQ'", S), (", 'ADAS11'", S), (',', D)],
            [('    # Demografico + Genetico', C)],
            [("    'AGE'", S), (", 'APOE4'", S), (", 'PTEDUCAT'", S), (',', D)],
            [('    # Volumetria MRI normalizada por ICV (FreeSurfer)', C)],
            [("    'Hippocampus'", S), (", 'Entorhinal'", S),
             (", 'MidTemp'", S), (", 'Ventricles'", S), (',', D)],
            [('    # Biomarcadores LCR (Marco ATN)', C)],
            [("    'ABETA'", S), (", 'TAU'", S), (", 'PTAU'", S)],
            [(']', D)],
            [],
            [('TARGET_COLUMN = ', D), ("'DX'", S),
             ('   ', D), ('# Clases: CN=0, MCI=1, AD=2', C)],
            [],
            [('# Distribucion por dominio ATN:', C)],
            [('#   A — Amiloide   : ABETA                              (1 feature)', C)],
            [('#   T — Tau        : TAU, PTAU                          (2 features)', C)],
            [('#   N — Volumetria : Hippocampus, Entorhinal, MidTemp, Ventricles  (4 features)', C)],
            [('#   Cognitivo      : MMSE, CDR, FAQ, ADAS11             (4 features)', C)],
            [('#   Demografico    : AGE, APOE4, PTEDUCAT               (3 features)', C)],
        ]
    },

    # ── 11.1 ── Estrategia dos fases ──────────────────────────────────────────
    {
        'title': '11.1.1  Entrenamiento CNN Dual-Backbone — Estrategia en Dos Fases (Transfer Learning)',
        'fname': 'codigo_11_1_dos_fases.png',
        'lines': [
            [('# ── FASE 1: Pre-entrenamiento (epochs 1-20) ─────────────────────', C)],
            [('# Congelar el backbone, solo entrenar el clasificador', C)],
            [('for', K), (' param ', D), ('in', K), (' model.resnet_features.parameters():', D)],
            [('    param.requires_grad = ', D), ('False', K)],
            [('for', K), (' param ', D), ('in', K), (' model.densenet_features.parameters():', D)],
            [('    param.requires_grad = ', D), ('False', K)],
            [],
            [('optimizer = optim.Adam(model.classifier.parameters(), lr=', D), ('1e-3', N), (')', D)],
            [],
            [('# ── FASE 2: Fine-tuning (epochs 21-100) ─────────────────────────', C)],
            [('# Descongelar todo el modelo con LR reducido', C)],
            [('for', K), (' param ', D), ('in', K), (' model.parameters():', D)],
            [('    param.requires_grad = ', D), ('True', K)],
            [],
            [('optimizer = optim.AdamW(model.parameters(), lr=', D), ('1e-4', N),
             (', weight_decay=', D), ('0.05', N), (')', D)],
            [('scheduler = torch.optim.lr_scheduler.OneCycleLR(', D)],
            [('    optimizer,', D)],
            [('    max_lr=', D), ('1e-4', N), (',', D)],
            [('    steps_per_epoch=len(train_loader),', D)],
            [('    epochs=', D), ('80', N), (',', D)],
            [('    pct_start=', D), ('0.3', N),
             ('        ', D), ('# 30% warmup linear + 70% decay coseno', C)],
            [(')', D)],
            [('criterion = nn.CrossEntropyLoss(label_smoothing=', D), ('0.1', N), (')', D)],
        ]
    },

    # ── 11.1.2 ── Loop de Entrenamiento ──────────────────────────────────────
    {
        'title': '11.1.2  Loop de Entrenamiento Principal — AMP FP16 + Gradient Clipping',
        'fname': 'codigo_11_2_train_loop.png',
        'lines': [
            [('def', K), (' train_epoch', F),
             ('(model, loader, optimizer, criterion, scheduler, device):', D)],
            [('    model.train()', D)],
            [('    running_loss, correct, total = ', D),
             ('0.0', N), (', ', D), ('0', N), (', ', D), ('0', N)],
            [],
            [('    for', K), (' images, labels ', D), ('in', K), (' loader:', D)],
            [('        images, labels = images.to(device), labels.to(device)', D)],
            [('        optimizer.zero_grad()', D)],
            [],
            [('        # Mixed precision FP16 → reduce VRAM ~50%', C)],
            [('        with', K), (' torch.cuda.amp.autocast():', D)],
            [('            outputs = model(images)', D)],
            [('            loss    = criterion(outputs, labels)', D)],
            [],
            [('        scaler.scale(loss).backward()', D)],
            [('        scaler.unscale_(optimizer)', D)],
            [('        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=', D), ('1.0', N), (')', D)],
            [('        scaler.step(optimizer)', D)],
            [('        scaler.update()', D)],
            [('        scheduler.step()  ', D), ('# step por batch (OneCycleLR)', C)],
            [],
            [('        running_loss += loss.item()', D)],
            [('        _, predicted  = outputs.max(', D), ('1', N), (')', D)],
            [('        correct       += predicted.eq(labels).sum().item()', D)],
            [('        total         += labels.size(', D), ('0', N), (')', D)],
            [],
            [('    return', K), (' running_loss / len(loader),  ', D),
             ('100.', N), (' * correct / total', D)],
        ]
    },

    # ── 11.2 ── Optuna XGBoost ────────────────────────────────────────────────
    {
        'title': '11.2  Optimización de Hiperparámetros XGBoost con Optuna TPE (100 Trials)',
        'fname': 'codigo_11_3_optuna_xgboost.png',
        'lines': [
            [('import', K), (' optuna', F)],
            [('from', K), (' sklearn.model_selection ', D), ('import', K), (' cross_val_score', F)],
            [],
            [('def', K), (' objective', F), ('(trial):', D)],
            [('    params = {', D)],
            [("        'n_estimators'    : trial.suggest_int('n_estimators', ", D),
             ('200', N), (', ', D), ('1200', N), ('),', D)],
            [("        'max_depth'       : trial.suggest_int('max_depth', ", D),
             ('3', N), (', ', D), ('8', N),    ('),', D)],
            [("        'learning_rate'   : trial.suggest_float('learning_rate', ", D),
             ('0.01', N), (', ', D), ('0.2', N), (', log=', D), ('True', K), ('),', D)],
            [("        'subsample'       : trial.suggest_float('subsample', ", D),
             ('0.6', N), (', ', D), ('1.0', N), ('),', D)],
            [("        'colsample_bytree': trial.suggest_float('colsample_bytree', ", D),
             ('0.5', N), (', ', D), ('1.0', N), ('),', D)],
            [("        'reg_alpha'       : trial.suggest_float('reg_alpha', ", D),
             ('1e-3', N), (', ', D), ('10.0', N), (', log=', D), ('True', K), ('),', D)],
            [("        'reg_lambda'      : trial.suggest_float('reg_lambda', ", D),
             ('1e-3', N), (', ', D), ('10.0', N), (', log=', D), ('True', K), ('),', D)],
            [("        'min_child_weight': trial.suggest_int('min_child_weight', ", D),
             ('1', N), (', ', D), ('10', N), ('),', D)],
            [("        'gamma'           : trial.suggest_float('gamma', ", D),
             ('0', N), (', ', D), ('0.5', N), ('),', D)],
            [("        'tree_method': 'hist',  'device': 'cuda',  'random_state': ", D), ('42', N)],
            [('    }', D)],
            [('    model  = xgb.XGBClassifier(**params)', D)],
            [('    scores = cross_val_score(model, X_train, y_train, cv=', D),
             ('5', N), (", scoring='accuracy', n_jobs=-", D), ('1', N), (')', D)],
            [('    return', K), (' scores.mean()', D)],
            [],
            [('study = optuna.create_study(direction=', D), ("'maximize'", S), (',', D)],
            [('                    sampler=optuna.samplers.TPESampler(seed=', D), ('42', N), ('))', D)],
            [('study.optimize(objective, n_trials=', D), ('100', N),
             (', timeout=', D), ('3600', N), (')', D)],
            [('best_params = study.best_params  ', D), ('# Best CV-5 accuracy: 0.870', C)],
        ]
    },

    # ── 11.3 ── Monitorización ────────────────────────────────────────────────
    {
        'title': '11.3  Monitorización del Entrenamiento — Sistema de Reporte en Tiempo Real',
        'fname': 'codigo_11_4_monitorizacion.png',
        'lines': [
            [('import', K), (' json, time', F)],
            [],
            [('def', K), (' update_realtime_stats', F),
             ('(epoch, train_loss, val_loss, train_acc, val_acc):', D)],
            [('    stats = {', D)],
            [("        'epoch'      : epoch,", D)],
            [("        'timestamp'  : time.strftime(", D), ("'%H:%M:%S'", S), ('),', D)],
            [("        'train_loss' : round(train_loss, ", D), ('4', N), ('),', D)],
            [("        'val_loss'   : round(val_loss,   ", D), ('4', N), ('),', D)],
            [("        'train_acc'  : round(train_acc,  ", D), ('2', N), ('),', D)],
            [("        'val_acc'    : round(val_acc,    ", D), ('2', N), ('),', D)],
            [("        'gpu_temp'   : get_gpu_temperature(),", D)],
            [("        'gpu_memory' : get_gpu_memory_usage()", D)],
            [('    }', D)],
            [("    with", K), (" open(", D), ("'benchmark_realtime_stats.json'", S),
             (", 'w'", S), (") as", K), (" f:", D)],
            [('        json.dump(stats, f, indent=', D), ('2', N), (')', D)],
            [],
            [('# ── Metricas de hardware — sesion de entrenamiento ───────────', C)],
            [('#   GPU       : NVIDIA RTX 4070  (16 GB VRAM)', C)],
            [('#   Temp max  : 62 C  (umbral de seguridad: 65 C)', C)],
            [('#   VRAM peak : 11.4 / 16 GB  (71.25% utilizacion)', C)],
            [('#   Tiempo/ep : ~47 s  (batch_size=64)', C)],
            [('#   Total     : 100 epocas  →  ~1h 18 min', C)],
        ]
    },
]

# ── RENDERER ───────────────────────────────────────────────────────────────────
FS = 12.0
GW = 0.046

def render(block):
    lines   = block['lines']
    n       = len(lines)
    pad_top = 0.088
    fig_h   = max(3.0, n * 0.55 + 1.6)
    fig_w   = 15.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # Barra título
    ax.add_patch(FancyBboxPatch(
        (0, 1 - pad_top), 1, pad_top,
        boxstyle='square,pad=0', lw=0, facecolor=BAR,
        transform=ax.transAxes, clip_on=False))
    for xi, col in [(0.016, TP), (0.036, TY), (0.056, TG)]:
        ax.add_patch(plt.Circle((xi, 1 - pad_top / 2), 0.007,
                                color=col, transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 1 - pad_top / 2, block['title'],
            transform=ax.transAxes, ha='center', va='center',
            fontsize=10.0, color='#CDD6F4', fontweight='bold', fontfamily='monospace')

    # Gutter
    ax.add_patch(FancyBboxPatch(
        (0, 0), GW, 1 - pad_top,
        boxstyle='square,pad=0', lw=0, facecolor=GUT,
        transform=ax.transAxes, clip_on=False))
    ax.plot([GW, GW], [0, 1 - pad_top], color=BO, lw=0.7, transform=ax.transAxes)

    content_h = 1.0 - pad_top - 0.024
    step = content_h / max(n, 1)

    for i, toks in enumerate(lines):
        y = (1 - pad_top - 0.02) - i * step - step * 0.3
        ax.text(GW * 0.88, y, str(i + 1),
                transform=ax.transAxes, ha='right', va='center',
                fontsize=8.5, color=LN, fontfamily='monospace')
        if not toks:
            continue
        x = GW + 0.012
        for tok in toks:
            if not isinstance(tok, tuple) or len(tok) != 2:
                continue
            txt, col = tok
            ax.text(x, y, txt,
                    transform=ax.transAxes, ha='left', va='center',
                    fontsize=FS, color=col, fontfamily='monospace')
            x += len(txt) * 0.008

    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, boxstyle='square,pad=0', lw=1.0,
        edgecolor=BO, facecolor='none',
        transform=ax.transAxes, clip_on=False))

    out_path = os.path.join(out_dir, block['fname'])
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=185, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    shutil.copy2(out_path, fig_dir)
    print(f"[OK] {block['fname']}")

for b in BLOCKS:
    render(b)

print(f"\nTotal: {len(BLOCKS)} imagenes generadas.")
