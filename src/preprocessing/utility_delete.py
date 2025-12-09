import os
from pathlib import Path

# --- CONFIGURATION ---
# Paths relative to this script
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

# Directories to scan
TARGET_DIRS = [
    DATA_ROOT / "train" / "images",
    DATA_ROOT / "validation" / "images",
    DATA_ROOT / "train" / "annotations",
    DATA_ROOT / "validation" / "annotations"
]

PREFIX = "aug_"

def clean_directory(directory):
    if not directory.exists():
        print(f"Skipping (not found): {directory}")
        return 0

    deleted_count = 0
    
    # Walk through directory (handles subfolders like 'crazing', 'patches' in images)
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.startswith(PREFIX):
                file_path = Path(root) / filename
                try:
                    os.remove(file_path)
                    # print(f"Deleted: {file_path.name}") # Uncomment for verbose output
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")
    
    return deleted_count

if __name__ == "__main__":
    print(f"Starting cleanup of files starting with '{PREFIX}'...")
    print(f"Scanning in: {DATA_ROOT}\n")

    total_deleted = 0

    for target_dir in TARGET_DIRS:
        print(f"Scanning: {target_dir} ...")
        count = clean_directory(target_dir)
        print(f"   -> Removed {count} files.")
        total_deleted += count

    print(f"\nCleanup Complete. Total files deleted: {total_deleted}")