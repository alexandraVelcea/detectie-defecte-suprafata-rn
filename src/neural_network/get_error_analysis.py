import os
import json
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# --- CONFIGURARE ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "defect_detector_ult" / "weights" / "best.pt"
TEST_IMG_DIR = PROJECT_ROOT / "data" / "test" / "images"
TEST_LBL_DIR = PROJECT_ROOT / "data" / "test" / "labels"
OUTPUT_JSON = PROJECT_ROOT / "results" / "error_analysis.json"

IOU_THRESHOLD = 0.5  # Pragul pentru a considera o suprapunere validă

# Clasele (trebuie să corespundă ordinii din data.yaml)
CLASSES = [
    'crazing', 'inclusion', 'patches', 
    'pitted_surface', 'rolled-in_scale', 'scratches'
]

def xywh2xyxy(x, w, h, img_w, img_h):
    """Conversie din format YOLO (center_x, center_y, w, h) normalizat în pixeli (x1, y1, x2, y2)."""
    x_c, y_c = x[0] * img_w, x[1] * img_h
    box_w, box_h = x[2] * img_w, x[3] * img_h
    
    x1 = x_c - box_w / 2
    y1 = y_c - box_h / 2
    x2 = x_c + box_w / 2
    y2 = y_c + box_h / 2
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2):
    """Calculează Intersection over Union (IoU) între două cutii."""
    # box = [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def analyze_errors():
    print(f"--- STARTING ERROR ANALYSIS ---")
    print(f"Model: {MODEL_PATH}")
    
    # 1. Încarcă Modelul
    if not MODEL_PATH.exists():
        print("Error: Model not found. Please train first.")
        return
    model = YOLO(MODEL_PATH)

    error_logs = []
    
    # 2. Iterează prin imaginile de test
    test_images = list(TEST_IMG_DIR.glob("*.jpg")) + list(TEST_IMG_DIR.glob("*.png"))
    
    for img_path in tqdm(test_images, desc="Analyzing"):
        filename = img_path.name
        label_path = TEST_LBL_DIR / (img_path.stem + ".txt")
        
        # --- A. Citește Ground Truth (Real) ---
        gt_boxes = []
        gt_classes = []
        
        # Obține dimensiunile imaginii pentru denormalizare
        # Folosim predicția modelului doar pentru a citi dimensiunea rapid (sau cv2)
        # Putem face un predict rapid fără save
        results = model.predict(img_path, conf=0.25, verbose=False)
        r = results[0]
        img_h, img_w = r.orig_shape
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    cls_id = int(parts[0])
                    # Conversie la x1, y1, x2, y2
                    bbox = xywh2xyxy(parts[1:], parts[2], parts[3], img_w, img_h)
                    gt_boxes.append(bbox)
                    gt_classes.append(cls_id)

        # --- B. Obține Predicțiile ---
        pred_boxes = r.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        pred_classes = r.boxes.cls.cpu().numpy().astype(int)
        pred_confs = r.boxes.conf.cpu().numpy()
        
        # --- C. Logică de Match (GT vs Pred) ---
        # Ținem evidența care GT și care Pred au fost potrivite
        matched_gt = set()
        matched_pred = set()
        
        # Cazul 1: Căutăm potriviri pentru fiecare GT (Recall check)
        for i, gt_box in enumerate(gt_boxes):
            best_iou = 0
            best_match_idx = -1
            
            for j, p_box in enumerate(pred_boxes):
                iou = calculate_iou(gt_box, p_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = j
            
            # Verificăm dacă e Match Valid
            if best_iou >= IOU_THRESHOLD:
                matched_gt.add(i)
                matched_pred.add(best_match_idx)
                
                # Verificare Clasificare
                if gt_classes[i] != pred_classes[best_match_idx]:
                    # EROARE: Clasificare Greșită (Misclassification)
                    error_logs.append({
                        "filename": filename,
                        "type": "Misclassification",
                        "real_class": CLASSES[gt_classes[i]],
                        "predicted_class": CLASSES[pred_classes[best_match_idx]],
                        "confidence": float(pred_confs[best_match_idx]),
                        "iou": float(best_iou),
                        "cause": "Visual Similarity"
                    })
            else:
                # EROARE: Defect Ratat (False Negative)
                error_logs.append({
                    "filename": filename,
                    "type": "False Negative",
                    "real_class": CLASSES[gt_classes[i]],
                    "predicted_class": "Background",
                    "confidence": 0.0,
                    "iou": 0.0,
                    "cause": "Low Contrast / Small Object"
                })

        # Cazul 2: Căutăm Predicții care nu au GT (False Positives)
        for j, p_box in enumerate(pred_boxes):
            if j not in matched_pred:
                # EROARE: Alarmă Falsă (False Positive)
                error_logs.append({
                    "filename": filename,
                    "type": "False Positive",
                    "real_class": "Background",
                    "predicted_class": CLASSES[pred_classes[j]],
                    "confidence": float(pred_confs[j]),
                    "iou": 0.0,
                    "cause": "Noise / Texture Confusion"
                })

    # 3. Salvare JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(error_logs, f, indent=4)
        
    print(f"\nAnalysis Complete.")
    print(f"Errors found: {len(error_logs)}")
    print(f"Report saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    analyze_errors()