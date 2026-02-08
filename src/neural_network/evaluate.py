import pandas as pd
import shutil
import json
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 1. Source Directory (Where YOLO saved the training logs)
# Adjust 'defect_detector_HD' if you used a different name in train.py
TRAIN_RUN_DIR = PROJECT_ROOT / "models" / "surface_defect_model"
SOURCE_CSV = TRAIN_RUN_DIR / "results.csv"

# 2. Destination Directories
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

def export_history():
    print(f"Looking for training artifacts in: {TRAIN_RUN_DIR}")
    
    if not SOURCE_CSV.exists():
        print(f"Error: Could not find results.csv at {SOURCE_CSV}")
        return

    # Ensure destination folders exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # --- PART 1: CSV Export ---
        print("\nExporting Training History CSV...")
        df = pd.read_csv(SOURCE_CSV)
        # Clean column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]
        
        dest_csv = RESULTS_DIR / "training_history.csv"
        df.to_csv(dest_csv, index=False)
        print(f"   Saved to: {dest_csv}")

        # --- PART 2: JSON Metrics Export ---
        print("\nCreating test_metrics.json...")
        # We take the metrics from the LAST epoch (best representation of final model state)
        last_epoch = df.iloc[-1]
        
        # Mapping YOLO columns to your requirement
        metrics_data = {
            "epoch": int(last_epoch['epoch']),
            "train_box_loss": float(last_epoch['train/box_loss']),
            "train_cls_loss": float(last_epoch['train/cls_loss']),
            "val_box_loss": float(last_epoch['val/box_loss']),
            "val_cls_loss": float(last_epoch['val/cls_loss']),
            "metrics": {
                "precision": float(last_epoch['metrics/precision(B)']),
                "recall": float(last_epoch['metrics/recall(B)']),
                "mAP_50": float(last_epoch['metrics/mAP50(B)']),
                "mAP_50_95": float(last_epoch['metrics/mAP50-95(B)'])
            }
        }
        
        dest_json = RESULTS_DIR / "test_metrics.json"
        with open(dest_json, "w") as f:
            json.dump(metrics_data, f, indent=4)
        print(f"   Saved to: {dest_json}")

        # --- PART 3: Visuals Export (Docs) ---
        print("\nCopying Visuals to Docs...")
        
        # Define files to copy (Source Name -> Dest Name)
        files_to_copy = {
            "results.png": "loss_curve.png",
            "confusion_matrix.png": "confusion_matrix.png"
        }
        
        for src_name, dest_name in files_to_copy.items():
            src_file = TRAIN_RUN_DIR / src_name
            if src_file.exists():
                shutil.copy(src_file, DOCS_DIR / dest_name)
                print(f"   Copied {src_name} -> docs/{dest_name}")
            else:
                print(f"   Warning: Could not find {src_name} (skipping)")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    export_history()