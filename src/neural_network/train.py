from ultralytics import YOLO
from pathlib import Path

# --- CONFIGURATION ---
# Define path to the data.yaml created by the previous script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
YAML_PATH = PROJECT_ROOT / "data" / "data.yaml"

def train_model():
    print(f"Loading configuration from: {YAML_PATH}")
    
    # 1. Load the model (Nano version for speed)
    model = YOLO('yolov8n.pt') 
    
    # 2. Train
    results = model.train(
        data=str(YAML_PATH),
        epochs=100,             # Adjustable: 50-100 is standard
        imgsz=200,              # Must match the size of your images
        batch=16,               # Batch size (lower if GPU memory error)
        name='defect_detector', # Save results in runs/detect/defect_detector
        patience=15,            # Early stopping
        device=0,               # '0' for GPU, 'cpu' for CPU
        workers=0               # Fix for Windows process errors
    )
    
    print("\nTraining Complete.")
    print(f"   Best weights saved at: {PROJECT_ROOT}/runs/detect/defect_detector/weights/best.pt")

if __name__ == '__main__':
    train_model()