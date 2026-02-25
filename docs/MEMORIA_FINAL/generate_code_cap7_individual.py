"""
Genera los 8 bloques de codigo Python del Capitulo 7 como imagenes PNG individuales.
Cada imagen corresponde a un apartado concreto.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, shutil

out_dir = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(out_dir, '..', '..', 'reports', 'figures')
os.makedirs(fig_dir, exist_ok=True)

# ── PALETA ────────────────────────────────────────────────────────────────────
BG  = '#1E1E2E'; GUT = '#181825'; BAR = '#11111B'
D   = '#CDD6F4'; K   = '#CBA6F7'; F   = '#89B4FA'
S   = '#A6E3A1'; C   = '#6C7086'; N   = '#FAB387'
P   = '#F9E2AF'; T   = '#F38BA8'; LN  = '#45475A'
BO  = '#313244'; TP  = '#F38BA8'; TY  = '#F9E2AF'; TG  = '#A6E3A1'

# ── BLOQUES INDIVIDUALES ──────────────────────────────────────────────────────
BLOCKS = [

    # ── 7.1 Paso 1 ── DICOM → NIfTI ─────────────────────────────────────────
    {
        'title': '7.1  Paso 1 — Conversión DICOM → NIfTI',
        'fname': 'codigo_7_paso1_dicom_nifti.png',
        'lines': [
            [('import', K), (' dicom2nifti', F)],
            [('import', K), (' os', F)],
            [],
            [('def', K), (' convert_series_to_nifti', F),
             ('(dicom_dir: ', D), ('str', T), (', output_path: ', D), ('str', T), ('):', D)],
            [('    """', C)],
            [('    Convierte una serie DICOM (160-200 cortes T1) a un volumen NIfTI unificado.', C)],
            [('    Requiere que los cortes esten en carpeta ordenados por InstanceNumber.', C)],
            [('    """', C)],
            [('    try', K), (':', D)],
            [('        dicom2nifti', F), ('.convert_directory(', D)],
            [('            dicom_directory', P), (' = dicom_dir,', D)],
            [('            output_folder', P),   (' = output_path,', D)],
            [('            compression', P),     (' = ', D), ('True', K),  ((',   '), D), ('# Genera .nii.gz', C)],
            [('            reorient', P),        (' = ', D), ('True', K),              ('           # Reorienta a espacio RAS', C)],
            [('        )', D)],
            [('        print', F), ('(f"[OK] Convertido: {', D), ('os', F),
             ('.path.basename(dicom_dir)}")', D)],
            [('    except', K), (' Exception', T), (' as', K), (' e:', D)],
            [('        print', F), ('(f"[ERROR] {dicom_dir}: {e}")', D)],
            [],
            [('# Resultado: 135 volumenes T1-weighted convertidos (tasa exito 100%)', C)],
            [('# Tamano medio: ~280 MB sin comprimir  ->  ~45 MB (.nii.gz)', C)],
        ]
    },

    # ── 7.1 Paso 2 ── Reorientación RAS ─────────────────────────────────────
    {
        'title': '7.1  Paso 2 — Reorientación Anatómica al Espacio RAS',
        'fname': 'codigo_7_paso2_reorientacion_ras.png',
        'lines': [
            [('import', K), (' nibabel ', D), ('as', K), (' nib', F)],
            [('import', K), (' numpy  ', D),  ('as', K), (' np',  F)],
            [],
            [('def', K), (' reorient_to_ras', F),
             ('(nifti_path: ', D), ('str', T), (') -> ', D), ('nib.Nifti1Image', T), (':', D)],
            [('    """', C)],
            [('    Reorienta el volumen al espacio RAS estandar.', C)],
            [('    Eje X -> derecha, Y -> anterior, Z -> superior.', C)],
            [('    Critico para consistencia entre escaneres 1.5T y 3T.', C)],
            [('    """', C)],
            [('    img     = ', D), ('nib', F), ('.load(nifti_path)', D)],
            [('    ras_img = ', D), ('nib', F), ('.as_closest_canonical(img)', D)],
            [('    return', K), (' ras_img', D)],
            [],
            [('# Aplicado a los 135 volumenes del corpus ADNI+OASIS-3', C)],
            [('# Elimina inconsistencias entre fabricantes (Siemens, GE, Philips)', C)],
        ]
    },

    # ── 7.1 Paso 3 ── Normalización Z-Score ─────────────────────────────────
    {
        'title': '7.1  Paso 3 — Normalización de Intensidad Z-Score por Volumen',
        'fname': 'codigo_7_paso3_normalizacion_zscore.png',
        'lines': [
            [('# Formula: z_voxel = (x_voxel - mu_brain) / sigma_brain', C)],
            [('# mu y sigma calculados SOLO sobre voxels con senal (valor > 0)', C)],
            [],
            [('def', K), (' normalize_zscore', F),
             ('(volume: ', D), ('np.ndarray', T), (') -> ', D), ('np.ndarray', T), (':', D)],
            [('    """', C)],
            [('    Normalizacion Z-score sobre voxels con senal cerebral.', C)],
            [('    Resultado: media=0, std=1 en tejido cerebral.', C)],
            [('    Elimina bias inter-escaner (variabilidad 1.5T vs 3T ~ 300%).', C)],
            [('    """', C)],
            [('    mask   = volume > ', D), ('0', N)],
            [('    mean_v = volume[mask].mean()', D)],
            [('    std_v  = volume[mask].std()', D)],
            [],
            [('    normalized = ', D), ('np', F), ('.zeros_like(volume, dtype=', D), ('np', F), ('.float32)', D)],
            [('    normalized[mask] = (volume[mask] - mean_v) / (std_v + ', D), ('1e-8', N),
             (')  ', D), ('# Epsilon evita division por cero', C)],
            [('    return', K), (' normalized', D)],
        ]
    },

    # ── 7.1 Paso 4 ── Bounding Box ──────────────────────────────────────────
    {
        'title': '7.1  Paso 4 — Segmentación de Bounding Box (Recorte Cerebral)',
        'fname': 'codigo_7_paso4_bounding_box.png',
        'lines': [
            [('# Elimina el espacio vacio (aire) alrededor del craneo.', C)],
            [('# Centra la red neuronal en el tejido cerebral util.', C)],
            [],
            [('def', K), (' crop_to_brain', F),
             ('(volume: ', D), ('np.ndarray', T), (') -> ', D), ('np.ndarray', T), (':', D)],
            [('    """Recorta el volumen al bounding box del cerebro."""', C)],
            [('    nonzero = ', D), ('np', F), ('.argwhere(volume > ', D), ('0', N), (')', D)],
            [],
            [('    if', K), (' len(nonzero) == ', D), ('0', N), (':', D)],
            [('        return', K), (' volume', D), ('   # Volumen vacio, devolver sin cambios', C)],
            [],
            [('    min_coords = nonzero.min(axis=', D), ('0', N), (')', D)],
            [('    max_coords = nonzero.max(axis=', D), ('0', N), (') + ', D), ('1', N)],
            [],
            [('    return', K), (' volume[', D)],
            [('        min_coords[', D), ('0', N), (']:max_coords[', D), ('0', N), ('],', D)],
            [('        min_coords[', D), ('1', N), (']:max_coords[', D), ('1', N), ('],', D)],
            [('        min_coords[', D), ('2', N), (']:max_coords[', D), ('2', N), (']', D)],
            [('    ]', D)],
        ]
    },

    # ── 7.1 Paso 5 ── Resize 128³ ────────────────────────────────────────────
    {
        'title': '7.1  Paso 5 — Redimensionado Uniforme a 128×128×128 Voxeles',
        'fname': 'codigo_7_paso5_resize_128.png',
        'lines': [
            [('from', K), (' scipy.ndimage ', D), ('import', K), (' zoom', F)],
            [],
            [('def', K), (' resize_volume', F),
             ('(volume: ', D), ('np.ndarray', T),
             (', target=(', D), ('128', N), (', ', D), ('128', N), (', ', D), ('128', N), (')) -> ', D),
             ('np.ndarray', T), (':', D)],
            [('    """', C)],
            [('    Redimensiona el volumen cerebral a dimensiones estandar 128^3.', C)],
            [('    Interpolacion de orden 1 (bilineal) para preservar tejido.', C)],
            [('    Entrada tipica: ~180x220x180 → Salida: 128x128x128', C)],
            [('    """', C)],
            [('    factors = [t / s ', D), ('for', K), (' t, s ', D), ('in', K),
             (' zip(target, volume.shape)]', D)],
            [('    return', K), (' zoom', F),
             ('(volume, factors, order=', D), ('1', N), (')  ', D),
             ('# order=1 bilineal, order=0 nearest', C)],
            [],
            [('# Pipeline completo orquestado:', C)],
            [('# for dicom_dir in dicom_dirs:', C)],
            [('#     vol = load_nifti(dicom_dir)   # Paso 1', C)],
            [('#     vol = reorient_to_ras(vol)    # Paso 2', C)],
            [('#     vol = normalize_zscore(vol)   # Paso 3', C)],
            [('#     vol = crop_to_brain(vol)      # Paso 4', C)],
            [('#     vol = resize_volume(vol)      # Paso 5  → tensor 128^3', C)],
        ]
    },

    # ── 7.2.2 ── Imputación por clase ────────────────────────────────────────
    {
        'title': '7.2.2  Tratamiento de Valores Faltantes — Imputación por Clase Diagnóstica',
        'fname': 'codigo_7_2_2_imputacion.png',
        'lines': [
            [('from', K), (' sklearn.impute ', D), ('import', K), (' SimpleImputer', F)],
            [('import', K), (' pandas ', D), ('as', K), (' pd', F)],
            [],
            [('# Estrategia: mediana por clase (preserva distribucion por grupo)', C)],
            [('def', K), (' impute_by_class', F),
             ('(df: ', D), ('pd.DataFrame', T), (', target_col: ', D), ('str', T),
             (') -> ', D), ('pd.DataFrame', T), (':', D)],
            [('    """Imputa NaN con la mediana de la clase diagnostica correspondiente."""', C)],
            [('    for', K), (' cls ', D), ('in', K), (' df[target_col].unique():', D)],
            [('        mask    = df[target_col] == cls', D)],
            [('        medians = df[mask].median(numeric_only=', D), ('True', K), (')', D)],
            [('        df.loc[mask] = df.loc[mask].fillna(medians)', D)],
            [('    return', K), (' df', D)],
            [],
            [('# ── Tasa de missing resuelta ──────────────────────────────────', C)],
            [('#   MMSE           :  2.3%  missing  → Imputado por mediana de clase', C)],
            [('#   Hippocampus    :  5.1%  missing  → Imputado por mediana de clase', C)],
            [('#   TAU / ABETA    : 18.7%  missing  → Imputado + flag binario adicional', C)],
        ]
    },

    # ── 7.2.3 ── Escalado y Codificación ────────────────────────────────────
    {
        'title': '7.2.3  Escalado Estándar y Codificación de Variables',
        'fname': 'codigo_7_2_3_escalado_codificacion.png',
        'lines': [
            [('from', K), (' sklearn.preprocessing ', D), ('import', K),
             (' StandardScaler', F), (', ', D), ('LabelEncoder', F)],
            [('import', K), (' joblib', F)],
            [],
            [('# ── Variables continuas → Z-score estandar ────────────────────', C)],
            [('# Resultado: media=0, std=1 en conjunto de entrenamiento', C)],
            [('scaler   = ', D), ('StandardScaler', T), ('()', D)],
            [('X_scaled = scaler.fit_transform(X_train[continuous_cols])', D)],
            [('X_test_s = scaler.transform(X_test[continuous_cols])', D),
             ('  # Solo transform, NO fit', C)],
            [],
            [('# ── Codificacion de variables categoricas ───────────────────────', C)],
            [('# APOE4: ya es binaria (0/1), no requiere encoding adicional', C)],
            [('# Diagnostico (target): CN=0, MCI=1, AD=2', C)],
            [('le        = ', D), ('LabelEncoder', T), ('()', D)],
            [('y_encoded = le.fit_transform(y_train)', D)],
            [],
            [('# ── Persistencia para inferencia en produccion ──────────────────', C)],
            [('joblib', F), ('.dump(scaler, ', D), ("'models/scaler_biomarkers.pkl'", S), (')', D)],
            [('joblib', F), ('.dump(le,     ', D), ("'models/label_encoder.pkl'",     S), (')', D)],
        ]
    },

    # ── 7.3 ── Data Augmentation CNN ────────────────────────────────────────
    {
        'title': '7.3  Data Augmentation CNN 2D — Fase Benchmarking (No Modelo de Producción)',
        'fname': 'codigo_7_3_augmentation_cnn.png',
        'lines': [
            [('from', K), (' torchvision ', D), ('import', K), (' transforms', F)],
            [],
            [('def', K), (' get_train_transforms', F),
             ('(img_size=(', D), ('224', N), (', ', D), ('224', N), (')):', D)],
            [('    """', C)],
            [('    Augmentation clinicamente conservador para imagenes MRI 2D:', C)],
            [('    - Rotacion ±15°    : simula variacion de posicion del paciente', C)],
            [('    - Flip horizontal  : el cerebro es casi simetrico', C)],
            [('    - ColorJitter min  : variaciones de contraste inter-escaner', C)],
            [('    - NO flip vertical : anatomicamente incorrecto en MRI', C)],
            [('    """', C)],
            [('    return', K), (' transforms', F), ('.Compose([', D)],
            [('        transforms', F), ('.', D), ('Resize', T), ('(img_size),', D)],
            [('        transforms', F), ('.', D), ('RandomHorizontalFlip', T), ('(p=', D), ('0.5', N), ('),', D)],
            [('        transforms', F), ('.', D), ('RandomRotation', T), ('(degrees=', D), ('15', N), ('),', D)],
            [('        transforms', F), ('.', D), ('ColorJitter', T),
             ('(brightness=', D), ('0.1', N), (', contrast=', D), ('0.1', N), ('),', D)],
            [('        transforms', F), ('.', D), ('ToTensor', T), ('(),', D)],
            [('        transforms', F), ('.', D), ('Normalize', T), ('(', D)],
            [('            mean=[', D), ('0.485', N), (', ', D), ('0.456', N), (', ', D), ('0.406', N),
             ('],   ', D), ('# Estadisticas ImageNet', C)],
            [('            std =[', D), ('0.229', N), (', ', D), ('0.224', N), (', ', D), ('0.225', N), (']', D)],
            [('        )', D)],
            [('    ])', D)],
            [],
            [('def', K), (' get_val_transforms', F),
             ('(img_size=(', D), ('224', N), (', ', D), ('224', N), (')):', D)],
            [('    """Sin augmentation para validacion y test — solo resize + normalize."""', C)],
            [('    return', K), (' transforms', F), ('.Compose([', D)],
            [('        transforms', F), ('.', D), ('Resize', T), ('(img_size),', D)],
            [('        transforms', F), ('.', D), ('ToTensor', T), ('(),', D)],
            [('        transforms', F), ('.', D), ('Normalize', T),
             ('(mean=[', D), ('0.485', N), (', ', D), ('0.456', N), (', ', D), ('0.406', N),
             ('], std=[', D), ('0.229', N), (', ', D), ('0.224', N), (', ', D), ('0.225', N), ('])', D)],
            [('    ])', D)],
        ]
    },
]

# ── RENDERER ──────────────────────────────────────────────────────────────────
FS = 10.2
GW = 0.046

def render(block):
    lines   = block['lines']
    n       = len(lines)
    pad_top = 0.088
    fig_h   = max(2.8, n * 0.395 + 1.4)
    fig_w   = 14.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # Barra título
    ax.add_patch(FancyBboxPatch(
        (0, 1 - pad_top), 1, pad_top,
        boxstyle='square,pad=0', lw=0, facecolor=BAR,
        transform=ax.transAxes, clip_on=False))
    for xi, col in [(0.019, TP), (0.042, TY), (0.065, TG)]:
        ax.add_patch(plt.Circle((xi, 1 - pad_top / 2), 0.008,
                                color=col, transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 1 - pad_top / 2, block['title'],
            transform=ax.transAxes, ha='center', va='center',
            fontsize=9.2, color='#CDD6F4', fontweight='bold', fontfamily='monospace')

    # Gutter
    ax.add_patch(FancyBboxPatch(
        (0, 0), GW, 1 - pad_top,
        boxstyle='square,pad=0', lw=0, facecolor=GUT,
        transform=ax.transAxes, clip_on=False))
    ax.plot([GW, GW], [0, 1 - pad_top], color=BO, lw=0.7, transform=ax.transAxes)

    content_h = 1.0 - pad_top - 0.024
    step = content_h / max(n, 1)

    for i, toks in enumerate(lines):
        y = (1 - pad_top - 0.016) - i * step - step * 0.3
        ax.text(GW * 0.88, y, str(i + 1),
                transform=ax.transAxes, ha='right', va='center',
                fontsize=7.5, color=LN, fontfamily='monospace')
        if not toks:
            continue
        x = GW + 0.010
        for tok in toks:
            if len(tok) != 2:
                continue
            txt, col = tok
            ax.text(x, y, txt,
                    transform=ax.transAxes, ha='left', va='center',
                    fontsize=FS, color=col, fontfamily='monospace')
            x += len(txt) * 0.0073

    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, boxstyle='square,pad=0', lw=1.0,
        edgecolor=BO, facecolor='none',
        transform=ax.transAxes, clip_on=False))

    out_path = os.path.join(out_dir, block['fname'])
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    shutil.copy2(out_path, fig_dir)
    print(f"[OK] {block['fname']}")

for b in BLOCKS:
    render(b)

print(f"\nTotal: {len(BLOCKS)} imagenes generadas.")
