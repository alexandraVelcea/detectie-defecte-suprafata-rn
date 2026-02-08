from ultralytics import YOLO
from pathlib import Path
import sys

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
YAML_PATH = PROJECT_ROOT / "results" / "data.yaml"

# Define where you want the results saved
# Example: PROJECT_ROOT / "training_runs"
CUSTOM_OUTPUT_DIR = PROJECT_ROOT / "models"

def train_model():
    # 1. Verify the data file exists before starting
    if not YAML_PATH.exists():
        print(f"ERROR: Could not find data.yaml at: {YAML_PATH}")
        sys.exit(1)

    print(f"Loading configuration from: {YAML_PATH}")
    print(f"Saving results to: {CUSTOM_OUTPUT_DIR}")

    # Load the model
    model = YOLO('models/yolov8n.pt') 

    # Train
    model.train(
        data=str(YAML_PATH),
        project=str(CUSTOM_OUTPUT_DIR), 
        name="defect_detector_HD",      
        epochs=100,
        imgsz=832,
        batch=2,                
        patience=20,
        device=0,
        workers=0,              
        cache=False,
        amp=False,
        exist_ok=True  # <--- OPTIONAL: Overwrite folder instead of creating defect_detector_HD2, HD3...
    )
    
    print(f"\nTraining Complete. Results saved in: {CUSTOM_OUTPUT_DIR / 'defect_detector_HD'}")

if __name__ == '__main__':
    train_model()