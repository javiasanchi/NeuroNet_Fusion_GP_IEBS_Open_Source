"""
Genera las imagenes de codigo Python del Capitulo 7 (Preprocesamiento):
  - codigo_7_1_pipeline_mri.png       : Pasos 1-5 DICOM->NIfTI pipeline (fase investigacion)
  - codigo_7_2_pipeline_tabular.png   : Imputacion, escalado y codificacion tabular (produccion)
  - codigo_7_3_data_augmentation.png  : Augmentation CNN 2D (fase benchmarking)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, shutil

# ── PALETA ────────────────────────────────────────────────────────────────────
BG_EDITOR   = '#1E1E2E'
BG_GUTTER   = '#181825'
BG_TITLEBAR = '#11111B'
D  = '#CDD6F4'   # default
K  = '#CBA6F7'   # keyword
F  = '#89B4FA'   # func / method
S  = '#A6E3A1'   # string
C  = '#6C7086'   # comment
N  = '#FAB387'   # number
P  = '#F9E2AF'   # param / keyword-arg
T  = '#F38BA8'   # type / class
LN = '#45475A'   # line-number
BO = '#313244'   # border

BLOCKS = [

    # ── BLOQUE 7.1 ── Pipeline MRI (investigacion) ─────────────────────────
    {
        'title': 'Codigo 7.1 - Pipeline MRI 3D: Conversion, Normalizacion y Resize (Fase de Investigacion)',
        'fname': 'codigo_7_1_pipeline_mri.png',
        'lines': [
            [('# Paso 1 — Conversion DICOM → NIfTI', C)],
            [('import', K), (' dicom2nifti', F), (', ', D), ('os', F)],
            [],
            [('def', K), (' convert_series_to_nifti', F), ('(dicom_dir: ', D), ('str', T), (', output_path: ', D), ('str', T), ('):', D)],
            [('    """Convierte serie DICOM (160-200 cortes T1) a volumen NIfTI."""', C)],
            [('    dicom2nifti', F), ('.convert_directory(', D)],
            [('        dicom_directory', P), ('=dicom_dir,', D)],
            [('        output_folder', P),   ('=output_path,', D)],
            [('        compression', P),     ('=', D), ('True', K), (',   ', D), ('# Genera .nii.gz', C)],
            [('        reorient', P),        ('=', D), ('True', K),             ('        # Reorienta a RAS', C)],
            [('    )', D)],
            [],
            [('# Paso 2 — Reorientacion anatómica RAS', C)],
            [('import', K), (' nibabel ', D), ('as', K), (' nib', F)],
            [],
            [('def', K), (' reorient_to_ras', F), ('(nifti_path: ', D), ('str', T), (') → ', D), ('nib.Nifti1Image', T), (':', D)],
            [('    img     = ', D), ('nib', F), ('.load(nifti_path)', D)],
            [('    ras_img = ', D), ('nib', F), ('.as_closest_canonical(img)', D)],
            [('    return', K), (' ras_img', D)],
            [],
            [('# Paso 3 — Normalizacion Z-score sobre voxels con señal', C)],
            [('def', K), (' normalize_zscore', F), ('(volume: ', D), ('np.ndarray', T), (') → ', D), ('np.ndarray', T), (':', D)],
            [('    mask   = volume > ', D), ('0', N)],
            [('    mean_v = volume[mask].mean()', D)],
            [('    std_v  = volume[mask].std()', D)],
            [('    norm   = ', D), ('np', F), ('.zeros_like(volume, dtype=', D), ('np', F), ('.float32)', D)],
            [('    norm[mask] = (volume[mask] - mean_v) / (std_v + ', D), ('1e-8', N), (')', D)],
            [('    return', K), (' norm', D)],
            [],
            [('# Paso 4 — Bounding Box (elimina aire/fondo)', C)],
            [('def', K), (' crop_to_brain', F), ('(volume: ', D), ('np.ndarray', T), (') → ', D), ('np.ndarray', T), (':', D)],
            [('    nonzero    = ', D), ('np', F), ('.argwhere(volume > ', D), ('0', N), (')', D)],
            [('    min_c      = nonzero.min(axis=', D), ('0', N), (')', D)],
            [('    max_c      = nonzero.max(axis=', D), ('0', N), (') + ', D), ('1', N)],
            [('    return', K), (' volume[min_c[', D), ('0', N), (']:max_c[', D), ('0', N), ('], min_c[', D), ('1', N), (']:max_c[', D), ('1', N), ('], ...]', D)],
            [],
            [('# Paso 5 — Redimensionado uniforme 128×128×128', C)],
            [('from', K), (' scipy.ndimage ', D), ('import', K), (' zoom', F)],
            [],
            [('def', K), (' resize_volume', F), ('(volume, target=(', D), ('128', N), (',', D), ('128', N), (',', D), ('128', N), (')):', D)],
            [('    factors = [t/s ', D), ('for', K), (' t, s ', D), ('in', K), (' zip(target, volume.shape)]', D)],
            [('    return', K), (' zoom(volume, factors, order=', D), ('1', N), (')  ', D), ('# Interpolacion bilineal', C)],
        ]
    },

    # ── BLOQUE 7.2 ── Pipeline Tabular (produccion) ────────────────────────
    {
        'title': 'Codigo 7.2 - Preprocesamiento de Biomarcadores Tabulares (Modelo de Produccion)',
        'fname': 'codigo_7_2_pipeline_tabular.png',
        'lines': [
            [('# ── Tratamiento de Valores Faltantes ──────────────────────────────', C)],
            [('from', K), (' sklearn.impute ', D), ('import', K), (' SimpleImputer', F)],
            [('import', K), (' pandas ', D), ('as', K), (' pd', F)],
            [],
            [('def', K), (' impute_by_class', F), ('(df: ', D), ('pd.DataFrame', T), (', target_col: ', D), ('str', T), (') → ', D), ('pd.DataFrame', T), (':', D)],
            [('    """Imputa valores faltantes con la mediana de cada clase diagnostica."""', C)],
            [('    for', K), (' cls ', D), ('in', K), (' df[target_col].unique():', D)],
            [('        mask    = df[target_col] == cls', D)],
            [('        medians = df[mask].median(numeric_only=', D), ('True', K), (')', D)],
            [('        df.loc[mask] = df.loc[mask].fillna(medians)', D)],
            [('    return', K), (' df', D)],
            [],
            [('# Tasa de missing resuelta:', C)],
            [('#   MMSE           :  2.3%  → Imputado por mediana de clase', C)],
            [('#   Hippocampus    :  5.1%  → Imputado por mediana de clase', C)],
            [('#   TAU / ABETA    : 18.7%  → Imputado + flag binario', C)],
            [],
            [('# ── Escalado Estandar y Codificacion ──────────────────────────────', C)],
            [('from', K), (' sklearn.preprocessing ', D), ('import', K), (' StandardScaler', F), (', LabelEncoder', F)],
            [],
            [('# Variables continuas → Z-score estandar (media=0, std=1)', C)],
            [('scaler   = ', D), ('StandardScaler', T), ('()', D)],
            [('X_scaled = scaler.fit_transform(X_train[continuous_cols])', D)],
            [],
            [('# APOE4: ya es binaria (0/1), no requiere encoding adicional', C)],
            [('# Target: CN=0, MCI=1, AD=2', C)],
            [('le        = ', D), ('LabelEncoder', T), ('()', D)],
            [('y_encoded = le.fit_transform(y_train)', D)],
            [],
            [('# Guardar scaler para produccion (persistencia del modelo)', C)],
            [('import', K), (' joblib', F)],
            [('joblib', F), ('.dump(scaler, ', D), ("'models/scaler_biomarkers.pkl'", S), (')', D)],
        ]
    },

    # ── BLOQUE 7.3 ── Data Augmentation CNN 2D (benchmarking) ─────────────
    {
        'title': 'Codigo 7.3 - Data Augmentation CNN 2D (Fase Benchmarking — No Produccion)',
        'fname': 'codigo_7_3_data_augmentation.png',
        'lines': [
            [('# Data augmentation conservador para MRI 2D (dataset Kaggle benchmarking)', C)],
            [('from', K), (' torchvision ', D), ('import', K), (' transforms', F)],
            [],
            [('def', K), (' get_train_transforms', F), ('(img_size=(', D), ('224', N), (', ', D), ('224', N), (' )):', D)],
            [('    """', C)],
            [('    Augmentation clinicamente conservador para imagenes MRI:', C)],
            [('    - Rotacion ±15°  : simula variacion de posicion del paciente', C)],
            [('    - Flip horizontal : el cerebro es casi simetrico', C)],
            [('    - ColorJitter     : variaciones de contraste inter-escaner', C)],
            [('    - NO flip vertical: anatomicamente incorrecto en MRI', C)],
            [('    """', C)],
            [('    return', K), (' transforms', F), ('.', D), ('Compose', T), ('([', D)],
            [('        transforms', F), ('.', D), ('Resize', T), ('(img_size),', D)],
            [('        transforms', F), ('.', D), ('RandomHorizontalFlip', T), ('(p=', D), ('0.5', N), ('),', D)],
            [('        transforms', F), ('.', D), ('RandomRotation', T), ('(degrees=', D), ('15', N), ('),', D)],
            [('        transforms', F), ('.', D), ('ColorJitter', T), ('(brightness=', D), ('0.1', N), (', contrast=', D), ('0.1', N), ('),', D)],
            [('        transforms', F), ('.', D), ('ToTensor', T), ('(),', D)],
            [('        transforms', F), ('.', D), ('Normalize', T), ('(', D)],
            [('            mean=[', D), ('0.485', N), (', ', D), ('0.456', N), (', ', D), ('0.406', N), ('],', D), ('   # ImageNet stats', C)],
            [('            std =[', D), ('0.229', N), (', ', D), ('0.224', N), (', ', D), ('0.225', N), (']', D)],
            [('        )', D)],
            [('    ])', D)],
            [],
            [('def', K), (' get_val_transforms', F), ('(img_size=(', D), ('224', N), (', ', D), ('224', N), (')):', D)],
            [('    """Sin augmentation para validacion / test."""', C)],
            [('    return', K), (' transforms', F), ('.', D), ('Compose', T), ('([', D)],
            [('        transforms', F), ('.', D), ('Resize', T), ('(img_size), ', D), ('transforms', F), ('.', D), ('ToTensor', T), ('(),', D)],
            [('        transforms', F), ('.', D), ('Normalize', T), ('(mean=[', D), ('0.485', N), (', ', D), ('0.456', N), (', ', D), ('0.406', N), ('], std=[', D), ('0.229', N), (', ', D), ('0.224', N), (', ', D), ('0.225', N), ('])', D)],
            [('    ])', D)],
        ]
    },

    # ── BLOQUE 7.2.3 ── Escalado e ICV (produccion) ───────────────────────
    {
        'title': 'Codigo 7.2.3 - Pipeline de Escalado, Normalizacion ICV y Codificación',
        'fname': 'codigo_7_2_3_escalado_codificacion.png',
        'lines': [
            [('from', K), (' sklearn.preprocessing ', D), ('import', K), (' StandardScaler', F), (', LabelEncoder', F)],
            [('import', K), (' numpy ', D), ('as', K), (' np', F)],
            [],
            [('def', K), (' preprocess_tabular_features', F), ('(X_train, y_train):', D)],
            [('    # 1. Normalización Biológica (ICV)', C)],
            [('    vol_cols = ', D), ("['Hippocampus', 'Entorhinal', 'MidTemporal', 'Ventricles']", S)],
            [('    for', K), (' col ', D), ('in', K), (' vol_cols:', D)],
            [('        X_train[', D), ("f'{col}_Norm'", S), ('] = X_train[col] / X_train[', D), ("'ICV'", S), (']', D)],
            [],
            [('    # 2. Escalado Estadístico (Z-Score)', C)],
            [('    continuous_cols = [', D)],
            [('        ', D), ("'BCMMSE'", S), (', ', D), ("'BCFAQ'", S), (', ', D), ("'entry_age'", S), (', ', D), ("'PTEDUCAT'", S), (',', D)],
            [('        ', D), ("'Hippo_Norm'", S), (', ', D), ("'Ento_Norm'", S), (', ', D), ("'MidTemp_Norm'", S), (', ', D), ("'Vent_Norm'", S), (',', D)],
            [('        ', D), ("'ABETA'", S), (', ', D), ("'TAU'", S), (', ', D), ("'PTAU'", S)],
            [('    ]', D)],
            [('    scaler = ', D), ('StandardScaler', T), ('()', D)],
            [('    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])', D)],
            [],
            [('    # 3. Codificación de Etiquetas (Target)', C)],
            [('    le = ', D), ('LabelEncoder', T), ('()', D)],
            [('    y_encoded = le.fit_transform(y_train)', D)],
            [],
            [('    # Guardar objetos para inferencia en producción', C)],
            [('    import', K), (' joblib', F)],
            [('    joblib', F), ('.dump(scaler, ', D), ("'models/scaler_production.joblib'", S), (')', D)],
            [('    joblib', F), ('.dump(le, ', D), ("'models/label_encoder.joblib'", S), (')', D)],
            [],
            [('    return', K), (' X_train, y_encoded, le', D)],
        ]
    },
]

# ── RENDERER ──────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(out_dir, '..', '..', 'reports', 'figures')
os.makedirs(fig_dir, exist_ok=True)

FS  = 12.0  # Incrementamos fuente
GW  = 0.046

for block in BLOCKS:
    lines   = block['lines']
    n       = len(lines)
    pad_top = 0.065 # Reducido de 0.085
    # Reducimos factor de altura de 0.55 a 0.42 y el offset de 1.6 a 1.0
    fig_h   = max(3.0, n * 0.42 + 1.0)
    fig_w   = 14.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG_EDITOR)
    ax.set_facecolor(BG_EDITOR)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # Barra título
    ax.add_patch(FancyBboxPatch(
        (0, 1 - pad_top), 1, pad_top,
        boxstyle='square,pad=0', lw=0, facecolor=BG_TITLEBAR,
        transform=ax.transAxes, clip_on=False))
    for xi, col in [(0.018, '#F38BA8'), (0.040, '#F9E2AF'), (0.062, '#A6E3A1')]:
        ax.add_patch(plt.Circle((xi, 1 - pad_top / 2), 0.007,
                                color=col, transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 1 - pad_top / 2, block['title'],
            transform=ax.transAxes, ha='center', va='center',
            fontsize=10.0, color='#CDD6F4', fontweight='bold', fontfamily='monospace')

    # Gutter
    ax.add_patch(FancyBboxPatch(
        (0, 0), GW, 1 - pad_top,
        boxstyle='square,pad=0', lw=0, facecolor=BG_GUTTER,
        transform=ax.transAxes, clip_on=False))
    ax.plot([GW, GW], [0, 1 - pad_top], color=BO, lw=0.7, transform=ax.transAxes)

    # Líneas de código
    content_h = 1.0 - pad_top - 0.025
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
            if len(tok) != 2:
                continue
            txt, col = tok
            ax.text(x, y, txt,
                    transform=ax.transAxes, ha='left', va='center',
                    fontsize=FS, color=col, fontfamily='monospace')
            # Ajustamos factor de ancho de 0.0073 a 0.008
            x += len(txt) * 0.008

    # Borde exterior
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, boxstyle='square,pad=0', lw=1.0,
        edgecolor=BO, facecolor='none',
        transform=ax.transAxes, clip_on=False))

    out_path = os.path.join(out_dir, block['fname'])
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=185, bbox_inches='tight',
                facecolor=BG_EDITOR, edgecolor='none')
    plt.close()
    shutil.copy2(out_path, fig_dir)
    print(f"[OK] {block['fname']}")

print("\nTodos los bloques del Capitulo 7 generados.")
