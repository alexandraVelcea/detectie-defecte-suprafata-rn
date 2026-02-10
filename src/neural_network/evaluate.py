import pandas as pd
import shutil
import json
from pathlib import Path

# ---------- COD PENTRU GENERARE STATISTICI: ----------

# ---------- docs/results/loss_curve.png ----------

# ---------- docs/confusion_matrix.png ----------

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 1. Source Directory
TRAIN_RUN_DIR = PROJECT_ROOT / "models" / "defect_detector_ult"
SOURCE_CSV = TRAIN_RUN_DIR / "results.csv"

# 2. Destination Directories
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

def export_history():
    print("--- EXPORTING RESULTS ---")
    print(f"Source: {TRAIN_RUN_DIR}")
    
    if not SOURCE_CSV.exists():
        print(f"Error: Could not find results.csv at {SOURCE_CSV}")
        print("Did the training finish successfully?")
        return

    # Ensure destination folders exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # --- PART 1: CSV Export ---
        print("\n1. Processing Training History...")
        df = pd.read_csv(SOURCE_CSV)
        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        
        # Save clean CSV
        dest_csv = RESULTS_DIR / "training_history.csv"
        df.to_csv(dest_csv, index=False)
        print(f"   CSV saved to: {dest_csv}")

        # --- PART 2: Metrics Calculation & JSON ---
        print("\n2. Calculating Metrics...")
        # Get the LAST epoch
        last_epoch = df.iloc[-1]
        
        # Retrieve core metrics safely
        precision = float(last_epoch.get('metrics/precision(B)', 0))
        recall = float(last_epoch.get('metrics/recall(B)', 0))
        map50 = float(last_epoch.get('metrics/mAP50(B)', 0))
        map50_95 = float(last_epoch.get('metrics/mAP50-95(B)', 0))
        
        # --- CALCULATE F1 SCORE ---
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        # Construct JSON Data
        metrics_data = {
            "model_name": TRAIN_RUN_DIR.name,
            "epoch": int(last_epoch.get('epoch', 0)),
            "accuracy": map50,
            "f1_score": f1_score,
            "train_box_loss": float(last_epoch.get('train/box_loss', 0)),
            "train_cls_loss": float(last_epoch.get('train/cls_loss', 0)),
            "val_box_loss": float(last_epoch.get('val/box_loss', 0)),
            "val_cls_loss": float(last_epoch.get('val/cls_loss', 0)),
            "metrics": {
                "precision": precision,
                "recall": recall,
                "mAP_50": map50,
                "mAP_50_95": map50_95
            }
        }
        
        # Save JSON
        dest_json = RESULTS_DIR / "test_metrics.json"
        with open(dest_json, "w") as f:
            json.dump(metrics_data, f, indent=4)
        print(f"   JSON saved to: {dest_json}")

        # --- PART 3: Console Output ---
        print("\n" + "="*30)
        print("FINAL MODEL PERFORMANCE")
        print("="*30)
        print(f"Accuracy (mAP@50):  {map50:.2%}")
        print(f"F1 Score:           {f1_score:.4f}")
        print(f"Precision:          {precision:.4f}")
        print(f"Recall:             {recall:.4f}")
        print("="*30)

        # --- PART 4: Visuals Export (Docs) ---
        print("\n3. Copying Visuals to Docs...")
        
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