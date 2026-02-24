import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

data_path = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/data/master_biomarker_dataset.csv'
model_path = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project/models/biomarker_xgb_optimized.joblib'

df = pd.read_csv(data_path)
features = ['BCMMSE', 'BCCDR', 'BCFAQ', 'PTGENDER', 'PTEDUCAT', 'entry_age', 
            'APOE4_carrier', 'Hippocampus', 'ABETA', 'TAU', 'PTAU', 'Centiloid']

X = df[features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = joblib.load(model_path)
y_pred = model.predict(X_test)

print("📊 Distribución de Clases Real:")
print(y.value_counts())
print("\n📈 Precisión Total:", accuracy_score(y_test, y_pred))
print("\n📝 Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['CN', 'MCI', 'AD']))

print("\n🧩 Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
