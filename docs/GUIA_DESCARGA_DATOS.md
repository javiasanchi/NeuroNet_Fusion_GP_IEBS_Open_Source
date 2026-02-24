# Guía de Descarga de Datos Reales: ADNI y OASIS-3

## 📊 OASIS-3 (Open Access Series of Imaging Studies)

### URLs Principales:
- **Sitio Web Oficial**: https://www.oasis-brains.org
- **Portal de Datos (XNAT Central)**: https://central.xnat.org
- **Página de Datasets**: https://www.oasis-brains.org/#access

### Proceso de Acceso:

1. **Registro Inicial**:
   - Visita: https://www.oasis-brains.org
   - Haz clic en "Apply To Access OASIS Data"
   - Acepta los términos de uso (Creative Commons Attribution 4.0)

2. **Credenciales**:
   - Recibirás credenciales de acceso por email
   - Usuario y contraseña para XNAT Central

3. **Descarga de Datos**:
   - Accede a: https://central.xnat.org
   - Navega a "Browse Data" → "OASIS3"
   - Selecciona "Download Images"
   - Opciones disponibles:
     * Formato: BIDS, NIFTI
     * Modalidades: T1w, T2w, FLAIR, DTI, etc.

### Contenido del Dataset:
- **1,098 participantes** (normales + deterioro cognitivo)
- **2,000+ sesiones de MRI**
- **Datos clínicos pareados** (MMSE, CDR, biomarcadores)
- **Formatos**: NIFTI (.nii.gz), BIDS

---

## 🧠 ADNI (Alzheimer's Disease Neuroimaging Initiative)

### URLs Principales:
- **Portal Principal**: https://adni.loni.usc.edu
- **IDA (Image & Data Archive)**: https://ida.loni.usc.edu
- **Solicitud de Acceso**: https://adni.loni.usc.edu/data-samples/access-data/

### Proceso de Acceso:

1. **Registro**:
   - Completa el formulario en: https://adni.loni.usc.edu/data-samples/access-data/
   - Requiere afiliación institucional
   - Aprobación en 1-3 días hábiles

2. **Descarga**:
   - Accede a IDA: https://ida.loni.usc.edu
   - Navega a "Download" → "Image Collections"
   - Selecciona:
     * ADNI1, ADNI2, ADNI3, ADNIGO
     * Modalidad: MRI (T1-weighted)
     * Formato: NIFTI

3. **Datos Clínicos**:
   - Descarga ADNIMERGE.csv desde:
     https://ida.loni.usc.edu/pages/access/studyData.jsp
   - Contiene: Diagnóstico, MMSE, ADAS-Cog, biomarcadores

### Contenido del Dataset:
- **2,000+ participantes**
- **MRI 3D de alta resolución** (1.5T y 3T)
- **Datos longitudinales** (seguimiento multi-año)
- **ADNIMERGE.csv**: Tabla maestra con todas las variables clínicas

---

## 🔧 Scripts de Descarga Automatizada

### Para OASIS-3 (usando XNAT):
```bash
# Requiere: curl, credenciales XNAT
curl -u USERNAME:PASSWORD \
  "https://central.xnat.org/data/archive/projects/OASIS3/subjects/OAS30001/experiments/OAS30001_MR_d0129/scans/anat1/files" \
  -o oasis_scan.zip
```

### Para ADNI (usando IDA):
```bash
# Descarga masiva con IDA Downloader
# Disponible en: https://ida.loni.usc.edu/pages/access/search.jsp?project=ADNI
# Selecciona imágenes → "Add to Collection" → "Download"
```

---

## 📁 Estructura Recomendada para el Proyecto

```
proyecto_global_IEBS/
├── data/
│   ├── raw/
│   │   ├── adni/
│   │   │   ├── mri/          # Archivos .nii.gz de ADNI
│   │   │   └── ADNIMERGE.csv # Datos clínicos
│   │   └── oasis3/
│   │       ├── mri/          # Archivos .nii.gz de OASIS-3
│   │       └── clinical/     # CSVs de datos clínicos
│   └── processed/
│       ├── images/           # MRI convertidas a PNG/JPG
│       └── tabular_clean.csv # Datos clínicos procesados
```

---

## ⚠️ Consideraciones Importantes

1. **Tamaño de Descarga**:
   - ADNI completo: ~500 GB
   - OASIS-3 completo: ~300 GB
   - **Recomendación**: Descarga solo T1-weighted MRI inicialmente

2. **Formato de Archivos**:
   - Los MRI vienen en formato NIFTI (.nii.gz)
   - Necesitarás convertirlos a 2D (slices) para el modelo actual
   - Herramientas: `nibabel`, `nilearn`, `SimpleITK`

3. **Alineación de Datos**:
   - ADNIMERGE.csv tiene columnas: `PTID`, `VISCODE`, `DX` (diagnóstico)
   - OASIS-3 usa: `subject_id`, `session_id`, `dx1` (diagnóstico)
   - Necesitarás un script de mapeo para unificar formatos

---

## 🚀 Próximos Pasos Recomendados

1. **Descarga Inicial** (Prioridad Alta):
   - ADNI: 200 sujetos (50 CN, 50 MCI, 50 AD, 50 EMCI)
   - OASIS-3: 200 sujetos (distribución similar)

2. **Preprocesamiento**:
   - Convertir NIFTI → PNG (slice central del hipocampo)
   - Unificar ADNIMERGE.csv + OASIS clinical data

3. **Re-entrenamiento**:
   - Usar datos reales pareados (mismo paciente: MRI + clínica)
   - Esperar mejora significativa (objetivo: >70%)

---

## 📞 Contacto de Soporte

- **OASIS**: oasis@wustl.edu
- **ADNI**: adni-info@loni.usc.edu
