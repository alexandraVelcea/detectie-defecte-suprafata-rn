import yaml
import pandas as pd
from pathlib import Path

# ---------- COD PENTRU GENERARE STATISTICI: ----------

# ---------- results/optimization_experiments.csv ----------

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Key parameters we want to extract from args.yaml
PARAMS_TO_EXTRACT = [
    "epochs",
    "batch",
    "imgsz",
    "optimizer",
    "lr0",          # Initial Learning Rate
    "lrf",          # Final Learning Rate Fraction
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "cos_lr",       # Cosine LR Scheduler
    "box",          # Box Loss Gain
    "cls",          # Class Loss Gain
    "dfl",          # DFL Loss Gain
    "close_mosaic"  # Mosaic Augmentation Turn-off
]

def generate_params_csv():
    print(f"--- EXTRACTING MODEL PARAMETERS ---")
    print(f"Scanning: {MODELS_DIR}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_data = []

    # Iterate over all folders in models/
    if not MODELS_DIR.exists():
        print(f"Error: Models directory not found.")
        return

    for model_folder in MODELS_DIR.iterdir():
        if model_folder.is_dir():
            args_path = model_folder / "args.yaml"
            
            if args_path.exists():
                try:
                    with open(args_path, 'r') as f:
                        args = yaml.safe_load(f)
                    
                    # Start with Model Name
                    row = {"Model Name": model_folder.name}
                    
                    # Extract desired parameters
                    for param in PARAMS_TO_EXTRACT:
                        row[param] = args.get(param, "N/A")
                    
                    model_data.append(row)
                    print(f"Found args for: {model_folder.name}")
                    
                except Exception as e:
                    print(f"Error reading {args_path}: {e}")
            else:
                # Optional: Skip folders without args.yaml (might be empty or non-model folders)
                pass

    if not model_data:
        print("No models with 'args.yaml' found.")
        return

    # Create DataFrame and Save
    df = pd.DataFrame(model_data)
    
    # Sort by Model Name for tidiness
    df = df.sort_values(by="Model Name")
    
    csv_path = RESULTS_DIR / "optimization_experiments.csv"
    df.to_csv(csv_path, index=False)
    
    print("-" * 30)
    print(f"CSV Generated: {csv_path}")
    print(df.to_string(index=False)) # Print a preview to console

if __name__ == "__main__":
    generate_params_csv()