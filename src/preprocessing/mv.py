import os
import shutil
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Source Directories (Where the data lives now)
SRC_IMG_DIR   = PROJECT_ROOT / "data" / "train" / "images"
SRC_XML_DIR   = PROJECT_ROOT / "data" / "train" / "annotations"
SRC_LABEL_DIR = PROJECT_ROOT / "data" / "train" / "labels"  # YOLO TXT files

# Destination Directories (Where to create the backup)
DEST_IMG_DIR   = PROJECT_ROOT / "data" / "generated" / "images"
DEST_XML_DIR   = PROJECT_ROOT / "data" / "generated" / "annotations"
DEST_LABEL_DIR = PROJECT_ROOT / "data" / "generated" / "labels"

def copy_all_augmented_data():
    print(f"--- COPYING AUGMENTED DATA (Images + XML + TXT) ---")
    
    # 1. Create Destination Folders
    # 'parents=True' creates data/generated if missing
    for d in [DEST_IMG_DIR, DEST_XML_DIR, DEST_LABEL_DIR]:
        if not d.exists():
            print(f"   Creating folder: {d}")
            d.mkdir(parents=True, exist_ok=True)

    if not SRC_IMG_DIR.exists():
        print(f"Error: Source directory {SRC_IMG_DIR} does not exist.")
        return

    # 2. Find Augmented Files
    # We look for images starting with "aug_"
    all_files = os.listdir(SRC_IMG_DIR)
    aug_files = [f for f in all_files if f.startswith("aug_") and f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not aug_files:
        print("No augmented files found to copy.")
        return

    print(f"Found {len(aug_files)} augmented images. Copying...")

    count = 0
    copied_xml = 0
    copied_txt = 0

    for img_name in aug_files:
        try:
            # --- 1. Copy Image ---
            src_img = SRC_IMG_DIR / img_name
            dest_img = DEST_IMG_DIR / img_name
            shutil.copy(str(src_img), str(dest_img))

            # Base name for labels (e.g., aug_crazing_123)
            base_name = img_name.rsplit('.', 1)[0]

            # --- 2. Copy XML (Annotations) ---
            xml_name = base_name + ".xml"
            src_xml = SRC_XML_DIR / xml_name
            dest_xml = DEST_XML_DIR / xml_name
            
            if src_xml.exists():
                shutil.copy(str(src_xml), str(dest_xml))
                copied_xml += 1

            # --- 3. Copy TXT (YOLO Labels) ---
            txt_name = base_name + ".txt"
            src_txt = SRC_LABEL_DIR / txt_name
            dest_txt = DEST_LABEL_DIR / txt_name
            
            if src_txt.exists():
                shutil.copy(str(src_txt), str(dest_txt))
                copied_txt += 1
            
            count += 1
            
            if count % 50 == 0:
                print(f"   Processed {count} sets...")

        except Exception as e:
            print(f"Error copying {img_name}: {e}")

    print("-" * 30)
    print(f"COPY COMPLETE.")
    print(f"   Images Copied: {count}")
    print(f"   XMLs Copied:   {copied_xml}")
    print(f"   TXTs Copied:   {copied_txt}")
    print(f"   Destination:   {PROJECT_ROOT}/data/generated")

if __name__ == "__main__":
    copy_all_augmented_data()