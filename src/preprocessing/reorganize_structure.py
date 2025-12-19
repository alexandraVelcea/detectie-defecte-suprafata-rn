import os
import shutil
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(os.getcwd())
BASE_RAW = PROJECT_ROOT / "data" / "raw"

# Source Directories (Based on your screenshot)
SRC_BASE = BASE_RAW / "NEU-DET"

# Destinations
DEST_BASE = BASE_RAW

def move_files(source_folder, dest_folder, file_extensions):
    """
    Moves files with specific extensions from source to destination.
    """
    if not source_folder.exists():
        print(f"⚠️  Source not found: {source_folder}")
        return

    # Create destination
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Scanning {source_folder.name}...")
    
    count = 0
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(file_extensions):
                src_path = Path(root) / file
                dst_path = dest_folder / file
                
                if dst_path.exists():
                    print(f"   Skipping {file} (already exists)")
                    continue
                
                shutil.move(str(src_path), str(dst_path))
                count += 1

    print(f"✅ Moved {count} files to: {dest_folder}\n")

def run_reorganization():
    print("--- REORGANIZING DATASET ---\n")
    
    # 1. TRAIN: Move Images and Annotations
    move_files(
        source_folder=SRC_BASE / "train" / "images",
        dest_folder=DEST_BASE / "train" / "images",
        file_extensions=('.jpg', '.jpeg', '.png', '.bmp')
    )
    move_files(
        source_folder=SRC_BASE / "train" / "annotations",
        dest_folder=DEST_BASE / "train" / "annotations",
        file_extensions=('.xml', '.txt', '.json')
    )

    # 2. VALIDATION: Move Images and Annotations
    move_files(
        source_folder=SRC_BASE / "validation" / "images",
        dest_folder=DEST_BASE / "validation" / "images",
        file_extensions=('.jpg', '.jpeg', '.png', '.bmp')
    )
    move_files(
        source_folder=SRC_BASE / "validation" / "annotations",
        dest_folder=DEST_BASE / "validation" / "annotations",
        file_extensions=('.xml', '.txt', '.json')
    )
    
    print("--- DONE ---")
    print(f"Check your folders at: {DEST_BASE}")

if __name__ == "__main__":
    run_reorganization()