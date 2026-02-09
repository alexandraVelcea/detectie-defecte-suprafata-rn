from ultralytics import YOLO
from pathlib import Path
import sys

# --- CONFIGURARE ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
YAML_PATH = PROJECT_ROOT / "results" / "data_colab.yaml" # Asigură-te că e calea corectă!
OUTPUT_DIR = PROJECT_ROOT / "models"

# 1. UPGRADE MODEL: Trecem la 'Medium'
#    Este mult mai capabil să înțeleagă defectele subtile.
MODEL_TYPE = "yolov8m.pt" 

def train_ultimate():
    print(f"--- STARTING OPTIMIZED TRAINING ---")
    print(f"Model: {MODEL_TYPE} (Medium)")
    print(f"GPU VRAM: Target ~10-12 GB Usage")
    
    model = YOLO(MODEL_TYPE) 

    model.train(
        data=str(YAML_PATH),
        project=str(OUTPUT_DIR),
        name="defect_detector_ult",
        
        # --- 1. Durată și Rezoluție ---
        epochs=150,             # Mai multe epoci pentru modelul 'Medium' (învață mai lent dar mai bine)
        patience=40,            # Îi dăm timp să iasă din platouri
        imgsz=832,              # Rezoluție HD (Multiplu de 32) - Compromis perfect viteză/detaliu
        
        # --- 2. Optimizare Hardware (Pentru 15GB VRAM) ---
        batch=16,               # Putem duce Batch 16 pe Medium cu 15GB
        nbs=64,                 # Nominal Batch Size pentru stabilitate
        
        # --- 3. Hiperparametri Avansați (The Secret Sauce) ---
        optimizer='AdamW',      # Cel mai bun pentru convergență pe date complexe
        lr0=0.0005,             # Learning rate mai mic pentru modelul mare (previne oscilațiile)
        cos_lr=True,            # Scade fin la final pentru maximizarea mAP
        
        # --- 4. Augmentări "Agresive" (Boost F1 Score) ---
        # Acestea creează date noi "dificile" pentru a forța modelul să fie robust
        mosaic=1.0,             # Standard YOLO (4 imagini în 1)
        mixup=0.15,             # [NOU] Amestecă 2 imagini (15% șansă). Ajută enorm la generalizare.
        copy_paste=0.3,         # [NOU] Lipește defecte în locuri noi (30% șansă).
        degrees=10.0,           # [NOU] Rotește ușor imaginile (+/- 10 grade)
        fliplr=0.5,             # Flip Orizontal
        flipud=0.5,             # Flip Vertical (Defectele nu au "sus" sau "jos")
        
        # --- 5. Sistem ---
        device=0,
        workers=8,
        cache=False,            # False e mai sigur ca să nu umplem RAM-ul sistemului
        exist_ok=True,
        plots=True              # Generează grafice detaliate la final
    )
    
    print("\nTraining Complete.")

if __name__ == '__main__':
    train_ultimate()