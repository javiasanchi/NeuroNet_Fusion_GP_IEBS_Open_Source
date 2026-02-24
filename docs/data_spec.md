# Especificación de Datos y Diccionario de Variables

## 1. Fuentes de Datos
El dataset unificado combina tres fuentes principales:
- **ADNI**: Alzheimer's Disease Neuroimaging Initiative.
- **OASIS**: Open Access Series of Imaging Studies.
- **Kaggle (Synthetic/Public)**: Datos complementarios para robustez.

## 2. Variables Clínicas (Tabulares)

| Variable | Descripción | Rango/Valores |
| :--- | :--- | :--- |
| **ID** | Identificador único del paciente (RID en ADNI). | Alfanumérico |
| **Age** | Edad del paciente al momento del estudio. | 50 - 95+ años |
| **Gender** | Sexo biológico del paciente. | Male, Female |
| **MMSE** | Mini-Mental State Examination (Estado cognitivo). | 0 (Deterioro) - 30 (Normal) |
| **ADAS13** | Alzheimer's Disease Assessment Scale. | Puntuación numérica |
| **CDRSB** | Clinical Dementia Rating Sum of Boxes. | Puntuación numérica |
| **ABETA** | Amiloide-beta en fluido cerebroespinal. | pg/mL (Continuo) |
| **TAU** | Proteína Tau total. | pg/mL (Continuo) |
| **PTAU** | Proteína Tau fosforilada. | pg/mL (Continuo) |
| **Diagnosis** | Etiqueta objetivo (Target). | CN, MCI, AD |

## 3. Especificación de Imágenes MRI

- **Formato**: JPG (Procesado a partir de NIfTI/DICOM originales).
- **Proyección**: Corte axial representativo.
- **Canales**: 3 (RGB para compatibilidad con backbones preentrenados).
- **Resolución**: 224 x 224 píxeles.
- **Normalización**: Intensidades igualadas por histograma global.

## 4. Características Extraídas (Features)
- **Texturales**: 5 métricas de GLCM (Contraste, Homogeneidad, etc.) y 10 bins de histograma LBP.
- **Semánticas**: 1280 dimensiones extraídas de la penúltima capa de EfficientNet-B0.
