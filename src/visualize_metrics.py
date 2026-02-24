import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def generate_visualizations():
    print("🎨 Generando visualizaciones avanzadas para el proyecto...")
    
    # 1. Preparación de rutas y carga de datos
    BASE_DIR = 'E:/MACHINE LEARNING/proyecto_global_IEBS/Analytical_Biomarker_Project'
    IMG_DIR = os.path.join(BASE_DIR, 'results', 'visuals')
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
        
    data_path = os.path.join(BASE_DIR, 'data', 'master_biomarker_v2_normalized.csv')
    model_path = os.path.join(BASE_DIR, 'models', 'neuro_fusion_final_v1.joblib')
    
    df = pd.read_csv(data_path).dropna(subset=['target'])
    data_pkg = joblib.load(model_path)
    model = data_pkg['model']
    features = data_pkg['features']
    
    X = df[features]
    y = df['target'].astype(int)
    
    # Reproducir el split 80/20 del entrenamiento
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    
    class_names = ['CN (Sano)', 'MCI', 'AD (Alzheimer)']
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    # 2. Gráfica de Importancia de Variables
    plt.figure(figsize=(12, 6))
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    colors = ['#8EBAD9' if '_Norm' not in c else '#FF9F68' for c in importances.index]
    importances.plot(kind='barh', color=colors)
    plt.title('Importancia de Variables en NeuroNet-Fusion\n(Resaltado: Variables Normalizadas)')
    plt.xlabel('Peso Relativo')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'feature_importance.png'), dpi=300)
    print("   ✅ Importancia de variables guardada.")

    # 3. Matriz de Confusión
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión Normalizada')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'confusion_matrix.png'), dpi=300)
    print("   ✅ Matriz de confusión guardada.")

    # 4. Curvas ROC
    y_score = model.predict_proba(X_test)
    plt.figure(figsize=(10, 8))
    colors = ['navy', 'darkorange', 'red']
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'ROC {class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC Multiclase - Capacidad Discriminativa')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'roc_curves.png'), dpi=300)
    print("   ✅ Curvas ROC guardadas.")

    # 5. Visualización del impacto de la normalización (Violin plots)
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='target', y='Hippo_Norm', data=df, hue='target', palette='muted', legend=False)
    plt.xticks([0, 1, 2], class_names)
    plt.title('Distribución de Atrofia Hipocampal (Normalizada por ICV)\nsegún Estado Clínico')
    plt.ylabel('Volumen Normalizado')
    plt.xlabel('Estado del Paciente')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'hippo_dist_normalized.png'), dpi=300)
    print("   ✅ Violin plot de atrofia guardado.")

    print(f"\n📁 Proceso completado. Gráficas disponibles en: {IMG_DIR}")

if __name__ == "__main__":
    generate_visualizations()
