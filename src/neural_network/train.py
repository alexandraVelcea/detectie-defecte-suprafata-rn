from ultralytics import YOLO
from pathlib import Path
import sys

# --- CONFIGURATION ---
# Calculates the root folder relative to where this script is located
# Adjust .parent count depending on your folder structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
YAML_PATH = PROJECT_ROOT / "data" / "data.yaml"

def train_model():
    # 1. Verify the data file exists before starting
    if not YAML_PATH.exists():
        print(f"ERROR: Could not find data.yaml at: {YAML_PATH}")
        print("   Check your 'PROJECT_ROOT' depth or file location.")
        sys.exit(1)

    print(f"Loading configuration from: {YAML_PATH}")
    
    # Load the model
    model = YOLO('yolov8n.pt') 

    # Train
    model.train(
        data=str(YAML_PATH),    # <--- USE THE FIXED PATH VARIABLE HERE
        epochs=100,
        imgsz=832,
        batch=2,                # Very low. If you have a decent GPU, try 4 or 8.
        name="defect_detector_HD",
        patience=20,
        device=0,
        workers=0,              # Good for Windows stability
        cache=False,
        amp=False,
        exist_ok=True           # Overwrites the folder instead of creating ...HD2, ...HD3
    )
    
    print("\nTraining Complete.")

if __name__ == '__main__':
    # This guard is crucial for Windows to prevent infinite loops with multiprocessing
    train_model()