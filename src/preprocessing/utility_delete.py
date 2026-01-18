import os
from pathlib import Path

# --- CONFIGURATION ---
# Resolve paths relative to this script
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

# Define directories to clean (scans subfolders recursively)
DIRS_TO_CLEAN = [
    PROJECT_ROOT / "data" / "raw" / "train",
    PROJECT_ROOT / "data" / "raw" / "validation"
]

def delete_augmented_files():
    print("--- DELETING AUGMENTED DATA (aug_*) ---")
    
    deleted_count = 0
    
    for base_dir in DIRS_TO_CLEAN:
        if not base_dir.exists():
            print(f"Skipping {base_dir} (not found)")
            continue
            
        # Walk through all subdirectories (images, annotations, class folders)
        for root, dirs, files in os.walk(base_dir):
            for filename in files:
                # Check if file starts with "aug_"
                if filename.startswith("aug_"):
                    file_path = Path(root) / filename
                    
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        # Optional: Print every 100 deletions to show progress
                        if deleted_count % 100 == 0:
                            print(f"Deleted {deleted_count} files...")
                            
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")

    print(f"\nCleanup Complete.")
    print(f"   Total files deleted: {deleted_count}")

if __name__ == "__main__":
    # Safety check to prevent accidents
    print(f"Target directories: {[str(d) for d in DIRS_TO_CLEAN]}")
    confirm = input("Are you sure you want to delete ALL files starting with 'aug_'? (yes/no): ")
    
    if confirm.lower() == "yes":
        delete_augmented_files()
    else:
        print("Operation cancelled.")