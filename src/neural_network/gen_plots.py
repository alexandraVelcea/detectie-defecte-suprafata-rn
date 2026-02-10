import matplotlib
matplotlib.use('Agg')  # Backend non-interactiv (critic pentru Colab)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- COD PENTRU GENERARE STATISTICI: ----------

# ---------- docs/optimization/f1_comparison.png ----------

# ---------- docs/optimization/accuracy_comparison.png ----------

# --- CONFIGURARE ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "optimization"

# Asigură-te că folderul de output există
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Definim modelele de comparat (Nume Folder : Nume Afișat pe Grafic)
MODELS_TO_COMPARE = {
    "surface_defect_model": "Etapa 5 (Nano)",
    "defect_detector_ult": "Etapa 6 (Medium)"
}

def get_model_data(folder_name):
    """Citește results.csv și extrage cea mai bună epocă."""
    csv_path = MODELS_DIR / folder_name / "results.csv"
    
    if not csv_path.exists():
        print(f"Atenție: Nu am găsit {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
        # Curățare nume coloane (YOLO pune spații în csv)
        df.columns = [c.strip() for c in df.columns]

        # Găsim epoca cu cel mai bun mAP@50
        best_idx = df['metrics/mAP50(B)'].idxmax()
        best_row = df.iloc[best_idx]

        # Calculăm F1 Score (2 * P * R / (P + R))
        p = best_row.get('metrics/precision(B)', 0)
        r = best_row.get('metrics/recall(B)', 0)
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

        return {
            "name": MODELS_TO_COMPARE[folder_name],
            "accuracy": best_row['metrics/mAP50(B)'],
            "f1": f1,
            "df": df  # Păstrăm tot istoricul pentru curbele de învățare
        }
    except Exception as e:
        print(f"Eroare la citirea {folder_name}: {e}")
        return None

def plot_bar_comparison(data_list, metric_key, title, ylabel, color, filename):
    """Funcție generică pentru bar charts."""
    names = [d['name'] for d in data_list]
    values = [d[metric_key] for d in data_list]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, values, color=color, width=0.5, alpha=0.9)
    
    plt.title(title, fontsize=14, pad=15, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.ylim(0, 1.05) # Scală 0-1
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Adăugare etichete cu valori pe bare
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2%}' if metric_key == 'accuracy' else f'{height:.4f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Salvat: {save_path}")

def plot_learning_curves(df, model_name):
    """Generează curbele de Loss și Accuracy pentru cel mai bun model."""
    plt.figure(figsize=(14, 6))

    # 1. Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='#e67e22', linewidth=2)
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='#c0392b', linewidth=2, linestyle='--')
    plt.title(f'Learning Curve: Loss ({model_name})', fontsize=12, fontweight='bold')
    plt.xlabel('Epoca')
    plt.ylabel('Box Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50 (Accuracy)', color='#27ae60', linewidth=2)
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95 (Strict)', color='#2ecc71', linewidth=1.5, linestyle=':')
    plt.title(f'Learning Curve: Accuracy ({model_name})', fontsize=12, fontweight='bold')
    plt.xlabel('Epoca')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)

    save_path = OUTPUT_DIR / "learning_curves_best.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Salvat: {save_path}")

def main():
    print("--- GENERARE GRAFICE COMPARATIVE ---")
    
    # 1. Extrage datele
    data = []
    for folder in MODELS_TO_COMPARE.keys():
        res = get_model_data(folder)
        if res:
            data.append(res)
    
    if not data:
        print("Nu am găsit date valide.")
        return

    # 2. Generează Accuracy Comparison
    plot_bar_comparison(
        data, 
        'accuracy', 
        'Comparație Acuratețe (mAP@50)', 
        'Mean Average Precision @ 50', 
        ['#95a5a6', '#2ecc71'], # Gri pt vechi, Verde pt nou
        'accuracy_comparison.png'
    )

    # 3. Generează F1 Comparison
    plot_bar_comparison(
        data, 
        'f1', 
        'Comparație F1-Score', 
        'F1 Score (Balanță Precision-Recall)', 
        ['#95a5a6', '#3498db'], # Gri pt vechi, Albastru pt nou
        'f1_comparison.png'
    )

    # 4. Generează Learning Curves (Doar pentru cel mai nou model - ultimul din listă)
    best_model_data = data[-1]
    plot_learning_curves(best_model_data['df'], best_model_data['name'])

    print("\nProces finalizat cu succes!")

if __name__ == "__main__":
    main()