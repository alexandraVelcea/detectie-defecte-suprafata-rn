import os
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Target the main data directory. This covers:
# data/raw, data/train, data/validation, data/test, data/generated
DATA_DIR = PROJECT_ROOT / "data"

def delete_all_augmented_data():
    print(f"--- 🗑️  DELETING ALL AUGMENTED DATA (aug_*) ---")
    print(f"Target Scope: {DATA_DIR} (and all subfolders)")
    
    if not DATA_DIR.exists():
        print(f"❌ Error: Data directory {DATA_DIR} does not exist.")
        return

    # Safety Confirmation
    confirm = input("\n⚠️  WARNING: This will permanently delete ALL files starting with 'aug_' (images, xml, txt).\nAre you sure? (yes/no): ")
    if confirm.lower() != "yes":
        print("Operation cancelled.")
        return

    counts = {
        "images": 0,
        "xml": 0,
        "txt": 0,
        "other": 0
    }
    
    total_deleted = 0

    # Walk through the entire data directory tree
    for root, dirs, files in os.walk(DATA_DIR):
        for filename in files:
            # Check for the augmentation prefix
            if filename.startswith("aug_"):
                file_path = Path(root) / filename
                
                try:
                    # Categorize for reporting
                    if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                        counts["images"] += 1
                    elif filename.lower().endswith('.xml'):
                        counts["xml"] += 1
                    elif filename.lower().endswith('.txt'):
                        counts["txt"] += 1
                    else:
                        counts["other"] += 1

                    # DELETE THE FILE
                    os.remove(file_path)
                    total_deleted += 1

                    # Progress update
                    if total_deleted % 100 == 0:
                        print(f"   Deleted {total_deleted} files...")

                except Exception as e:
                    print(f"❌ Error deleting {filename}: {e}")

    print("-" * 30)
    print("✅ CLEANUP COMPLETE.")
    print(f"   Total Files Deleted: {total_deleted}")
    print(f"   Breakdown:")
    print(f"     - Images: {counts['images']}")
    print(f"     - XMLs:   {counts['xml']}")
    print(f"     - TXTs:   {counts['txt']}")
    if counts['other'] > 0:
        print(f"     - Others: {counts['other']}")

if __name__ == "__main__":
    delete_all_augmented_data()