import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from collections import Counter

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Path to your trained model
MODEL_PATH = PROJECT_ROOT / "models" / "surface_defect_model" / "weights" / "best.pt"
# Path to Test Images
TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "test" / "images"
# Base Output Path
BASE_OUTPUT_PATH = PROJECT_ROOT / "docs" / "screenshots" / "inference_stats.png"

# Class Names (Must match your data.yaml order)
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

def get_unique_path(path):
    """
    If path exists, appends a counter (_1, _2, etc.) to the filename
    to avoid overwriting.
    """
    if not path.exists():
        return path
    
    counter = 1
    while True:
        new_path = path.parent / f"{path.stem}_{counter}{path.suffix}"
        if not new_path.exists():
            return new_path
        counter += 1

def generate_stats():
    # 1. Load Model
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Extract Model Name from the path
    model_name = MODEL_PATH.parent.parent.name 
    print(f"Loading model '{model_name}' from: {MODEL_PATH}")
    
    model = YOLO(MODEL_PATH)

    # 2. Run Inference on Test Set
    print(f"Running inference on {TEST_IMAGES_DIR}...")
    results = model.predict(source=str(TEST_IMAGES_DIR), conf=0.25, save=False, stream=True, verbose=False)

    # 3. Collect Statistics
    class_counts = Counter()
    confidence_scores = []
    total_images = 0
    images_with_defects = 0

    print("Processing results...")
    for r in results:
        total_images += 1
        boxes = r.boxes
        if len(boxes) > 0:
            images_with_defects += 1
            
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Safe lookup for class name
            if cls_id < len(CLASS_NAMES):
                class_name = CLASS_NAMES[cls_id]
            else:
                class_name = model.names[cls_id] # Fallback to internal model names
            
            class_counts[class_name] += 1
            confidence_scores.append(conf)

    # 4. Prepare Data for Plotting
    final_counts = {name: class_counts.get(name, 0) for name in CLASS_NAMES}
    df_counts = pd.DataFrame(list(final_counts.items()), columns=['Defect Type', 'Count'])
    
    # 5. Generate Plots
    # Determine unique output path
    unique_output_path = get_unique_path(BASE_OUTPUT_PATH)
    print(f"Generating visualization at {unique_output_path}...")
    
    # Setup Figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Title
    plt.suptitle(f'Inference Statistics: {model_name}\n(Test Set N={total_images} Images)', fontsize=16, fontweight='bold')

    # Plot 1: Bar Chart
    sns.barplot(x='Defect Type', y='Count', data=df_counts, ax=ax1, hue='Defect Type', palette='viridis', legend=False)
    ax1.set_title('Total Detections per Class', fontsize=14)
    ax1.set_xlabel('Defect Class', fontsize=12)
    ax1.set_ylabel('Number of Detections', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels
    for p in ax1.patches:
        height = int(p.get_height())
        if height > 0:
            ax1.annotate(f'{height}', 
                         (p.get_x() + p.get_width() / 2., height), 
                         ha = 'center', va = 'center', 
                         xytext = (0, 9), 
                         textcoords = 'offset points',
                         fontweight='bold')

    # Plot 2: Histogram
    if confidence_scores:
        sns.histplot(confidence_scores, bins=20, kde=True, ax=ax2, color='#3498db')
        ax2.set_title('Detection Confidence Distribution', fontsize=14)
        ax2.set_xlabel('Confidence Score (0.0 - 1.0)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.axvline(x=0.25, color='r', linestyle='--', label='Threshold (0.25)')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No Detections Found", ha='center', va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.90]) 
    
    # Save
    unique_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(unique_output_path, dpi=300)
    plt.close()
    
    print("\nSuccess!")
    print(f"   - Model Used: {model_name}")
    print(f"   - Chart saved to: {unique_output_path}")

if __name__ == "__main__":
    generate_stats()