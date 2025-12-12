from ultralytics import YOLO
import os

def train_model():
    # 1. Load the model
    # 'yolov8n.pt' is the Nano model (fastest). 
    # Use 'yolov8m.pt' (Medium) if you have a powerful GPU and want higher accuracy.
    model = YOLO('yolov8n.pt') 
    
    # 2. Define path to yaml config
    # Ensure this path is correct relative to where you run this script
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../preprocessing/data.yaml'))

    print(f"Starting training using config: {yaml_path}")

    # 3. Train
    results = model.train(
        data=yaml_path,
        epochs=100,          # 50-100 epochs is usually good for this size dataset
        imgsz=200,          # Match your image size (from XML/augmentation script)
        batch=16,           # Reduce to 8 or 4 if you run out of GPU memory
        name='defect_detector', # Name of the run folder
        device='0'          # Use '0' for GPU, 'cpu' for CPU
        # workers=0         # Uncomment if you get "DataLoader" errors on Windows
    )
    
    print("Training Complete. Model saved in 'runs/detect/defect_detector/weights/best.pt'")

if __name__ == '__main__':
    train_model()