import os
import time
import requests
import random
from pathlib import Path

# --- 1. SETUP & CONFIGURATION ---

script_location = Path(__file__).resolve().parent
# Go up two levels to find the root
project_root = script_location.parent.parent

CLASSES = [
    'crazing', 'inclusion', 'patches', 
    'pitted_surface', 'rolled-in_scale', 'scratches'
]

# Specific visual descriptions for better accuracy
DEFECT_DESCRIPTIONS = {
    'crazing': 'network of fine spiderweb cracks, ceramic-like fracture pattern',
    'inclusion': 'dark foreign material embedded in metal, oxide inclusion, impurity',
    'patches': 'discrete irregular areas of surface discoloration, rust patch',
    'pitted_surface': 'small deep corrosion craters, tiny holes, rough texture',
    'rolled-in_scale': 'rough embedded iron oxide scale, flaky surface texture',
    'scratches': 'deep linear abrasions, sharp cut marks, metal gouges'
}

# Goal is 1 for train, 1 for validation
GENERATION_CONFIG = {
    'train': (1, Path('../../data/raw/train/images')),
    'validation': (1, Path('../../data/raw/validation/images'))
}

# --- 2. GENERATION LOGIC ---

def generate_single_image(class_name, split_name, save_folder, index):
    """
    Generates a single image using Pollinations.ai
    """
    base_url = "https://image.pollinations.ai/prompt/"
    
    visual_desc = DEFECT_DESCRIPTIONS.get(class_name, "industrial defect")
    
    prompt_text = (
        f"close-up macro photo of industrial steel metal surface with {class_name} defect, "
        f"{visual_desc}, realistic metallic texture, neutral lighting, black and white"
        f"high contrast, sharp focus"
    )
    
    seed = int(time.time()) + index + random.randint(1, 1000)
    
    # Dimensions set to 200x200
    final_url = f"{base_url}{prompt_text}?width=200&height=200&nologo=true&seed={seed}&enhance=false"
    
    filename = f"gen_pollinations_{split_name}_{class_name}_{seed}.png"
    save_path = save_folder / filename
    
    try:
        print(f"   Requesting: {class_name} ({index})")
        
        # INCREASED TIMEOUT TO 60 SECONDS
        response = requests.get(final_url, timeout=60)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"   Saved: {filename}")
            return True
        else:
            print(f"   Failed. Status code: {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        print("   Timeout Error: The server took too long to respond.")
        return False
    except Exception as e:
        print(f"   Connection Error: {e}")
        return False

# --- 3. MAIN EXECUTION LOOP ---

if __name__ == "__main__":
    print("Starting Data Augmentation (1 Train + 1 Val per class)...")
    
    for split_name, (goal_count, relative_base_path) in GENERATION_CONFIG.items():
        print(f"\n=== Processing split: {split_name.upper()} ===")
        
        # Resolve absolute path based on script location
        base_destination_path = (script_location / relative_base_path).resolve()

        for class_name in CLASSES:
            print(f"\n--- Class: {class_name} ---")
            target_folder = base_destination_path / class_name
            
            # Ensure folder exists
            if not target_folder.exists():
                target_folder.mkdir(parents=True, exist_ok=True)
                print(f"   Created folder: {target_folder}")

            count = 0
            failures = 0
            
            while count < goal_count:
                success = generate_single_image(
                    class_name, 
                    split_name, 
                    target_folder, 
                    count + 1
                )
                
                if success:
                    count += 1
                    failures = 0 
                    time.sleep(1) 
                else:
                    failures += 1
                    print("   Retrying in 5 seconds...")
                    time.sleep(5)
                    
                    if failures > 3:
                        print(f"   Skipping {class_name} due to repeated errors.")
                        break

    print("\nAugmented dataset generation complete.")