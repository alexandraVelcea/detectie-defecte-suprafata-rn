import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"

# Define paths to the final splits
TRAIN_DIR = DATA_DIR / "train" / "images"
VAL_DIR = DATA_DIR / "validation" / "images"
TEST_DIR = DATA_DIR / "test" / "images"

# CSV Output Path
CSV_PATH = DOCS_DIR / "data_statistics.csv"
# Graph Output Path
GRAPH_PATH = DOCS_DIR / "generated_vs_real.png"

def count_images(directory):
    """
    Counts real vs synthetic images in a directory.
    Assumption based on your augmentation script:
    - Synthetic images start with "aug_"
    - Real images do NOT start with "aug_"
    """
    if not directory.exists():
        return 0, 0
    
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    synthetic_count = sum(1 for f in files if f.startswith("aug_"))
    real_count = len(files) - synthetic_count
    
    return real_count, synthetic_count

def generate_csv(stats):
    """Generates the statistics CSV file."""
    print(f"Generating CSV at {CSV_PATH}...")
    
    with open(CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header
        writer.writerow(["Split", "Real Images", "Synthetic Images", "Total", "% Synthetic"])
        
        # Rows
        total_real = 0
        total_syn = 0
        
        for split, (real, syn) in stats.items():
            total = real + syn
            perc = (syn / total * 100) if total > 0 else 0
            writer.writerow([split, real, syn, total, f"{perc:.2f}%"])
            total_real += real
            total_syn += syn
            
        # Total Row
        grand_total = total_real + total_syn
        grand_perc = (total_syn / grand_total * 100) if grand_total > 0 else 0
        writer.writerow(["TOTAL", total_real, total_syn, grand_total, f"{grand_perc:.2f}%"])

def generate_graph(stats):
    """Generates a stacked bar chart comparing Real vs Synthetic."""
    print(f"Generating Graph at {GRAPH_PATH}...")
    
    splits = list(stats.keys())
    real_counts = [stats[s][0] for s in splits]
    syn_counts = [stats[s][1] for s in splits]
    
    x = np.arange(len(splits))
    width = 0.5
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create stacked bars
    p1 = ax.bar(x, real_counts, width, label='Real (Original)', color='#3498db', edgecolor='black')
    p2 = ax.bar(x, syn_counts, width, bottom=real_counts, label='Synthetic (Augmented)', color='#e67e22', edgecolor='black')
    
    # Labels and Title
    ax.set_ylabel('Number of Images')
    ax.set_title('Contribution of Synthetic Data to Dataset Splits')
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.legend()
    
    # Add counts inside bars
    for i, (r, s) in enumerate(zip(real_counts, syn_counts)):
        if r > 0:
            ax.text(i, r/2, str(r), ha='center', va='center', color='white', fontweight='bold')
        if s > 0:
            ax.text(i, r + s/2, str(s), ha='center', va='center', color='white', fontweight='bold')
        # Total on top
        ax.text(i, r + s + 5, f"Total: {r+s}", ha='center', va='bottom', fontsize=9)

    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save
    plt.tight_layout()
    plt.savefig(GRAPH_PATH, dpi=300)
    plt.close()

def main():
    # Ensure docs directory exists
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Gather Statistics
    stats = {
        "train": count_images(TRAIN_DIR),
        "validation": count_images(VAL_DIR),
        "test": count_images(TEST_DIR)
    }
    
    # Debug print
    for split, (r, s) in stats.items():
        print(f"Found {split}: {r} Real, {s} Synthetic")

    # 2. Generate Files
    generate_csv(stats)
    generate_graph(stats)
    
    print("\n Statistics generation complete!")
    print(f"CSV: {CSV_PATH}")
    print(f"Graph: {GRAPH_PATH}")

if __name__ == "__main__":
    main()