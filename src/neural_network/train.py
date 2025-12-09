from ultralytics import YOLO

def train_model():
    # 1. Load a pre-trained model (transfer learning)
    # 'yolov8n.pt' will download automatically if you don't have it
    model = YOLO('yolov8n.pt') 

    # 2. Train the model
    results = model.train(
        data='src/preprocessing/data.yaml', # Path to your yaml config
        epochs=50,                         # 50 passes through the data
        imgsz=640,                         # Resize images to 640x640
        batch=16,                          # Number of images per batch
        name='surface_defect_model',       # Name of the project folder
        device='0'                         # Use '0' for GPU, 'cpu' for CPU
    )

if __name__ == '__main__':
    train_model()