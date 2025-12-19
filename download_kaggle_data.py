import os
import json
from pathlib import Path

# --- CONFIGURATION ---
# UPDATED: Using the exact slug from your link
DATASET_SLUG = "kaustubhdikshit/neu-surface-defect-database" 

# Look in the current working directory
PROJECT_ROOT = Path(os.getcwd())
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
CREDENTIALS_FILE = PROJECT_ROOT / "kaggle.json"

def setup_kaggle_credentials():
    print(f"Looking for credentials at: {CREDENTIALS_FILE}")
    
    if not CREDENTIALS_FILE.exists():
        print(f"Error: kaggle.json not found!")
        print(f"   Please ensure 'kaggle.json' is inside: {PROJECT_ROOT}")
        return False

    with open(CREDENTIALS_FILE, 'r') as f:
        data = json.load(f)
        
    os.environ['KAGGLE_USERNAME'] = data['username']
    os.environ['KAGGLE_KEY'] = data['key']
    print("Credentials loaded.")
    return True

def download_raw_only():
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    print("Authenticating with Kaggle...")
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("   Solution: Go to Kaggle Settings -> Expire API Token -> Create New Token.")
        return

    if not DATA_RAW_DIR.exists():
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset '{DATASET_SLUG}' to {DATA_RAW_DIR}...")
    
    try:
        api.dataset_download_files(DATASET_SLUG, path=DATA_RAW_DIR, unzip=True)
        
        # Cleanup zip files
        for item in os.listdir(DATA_RAW_DIR):
            if item.endswith(".zip"):
                os.remove(DATA_RAW_DIR / item)
                
        print(f"Download Complete! Files are in: {DATA_RAW_DIR}")
        
    except Exception as e:
        print(f"Error during download: {e}")
        if "403" in str(e):
            print("   This usually means your API Key is expired. Please generate a new kaggle.json.")

if __name__ == "__main__":
    if setup_kaggle_credentials():
        download_raw_only()