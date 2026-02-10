import os
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# Class names (Must match exactly the order used in your dataset)
CLASSES = [
    'crazing', 'inclusion', 'patches', 
    'pitted_surface', 'rolled-in_scale', 'scratches'
]

def convert_box(size, box):
    """
    Converts Pascal VOC (xmin, ymin, xmax, ymax) to YOLO (x_center, y_center, w, h).
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    
    # Calculate center x, center y, width, height
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    
    # Normalize (0 to 1)
    return (x * dw, y * dh, w * dw, h * dh)

def process_folder(split_name):
    """
    1. Reads XMLs from data/{split_name}/annotations
    2. Creates data/{split_name}/labels
    3. Writes YOLO TXT files into the new labels folder
    """
    # Define paths
    img_dir = DATA_ROOT / split_name / "images"
    xml_dir = DATA_ROOT / split_name / "annotations"
    label_dir = DATA_ROOT / split_name / "labels"  # <--- The target folder
    
    if not xml_dir.exists():
        print(f"Skipping {split_name}: 'annotations' folder not found.")
        return

    # --- CREATE THE LABELS FOLDER ---
    # parents=True ensures 'data/train' exists, exist_ok=True prevents error if 'labels' exists
    label_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Processing {split_name} ---")
    print(f"   Source: {xml_dir}")
    print(f"   Target: {label_dir}")
    
    count = 0
    empty_count = 0
    
    # Iterate over all XML files
    for xml_file in xml_dir.glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get Image Size
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            yolo_lines = []
            
            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                
                if cls_name not in CLASSES:
                    continue
                
                cls_id = CLASSES.index(cls_name)
                xmlbox = obj.find('bndbox')
                
                # Extract coordinates
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                
                # Convert to YOLO format
                bb = convert_box((w, h), b)
                
                # Format: class_id x_center y_center width height
                yolo_lines.append(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")

            # Define TXT filename (same basename as XML)
            txt_filename = xml_file.stem + ".txt"
            
            # Write to the labels folder
            with open(label_dir / txt_filename, 'w') as f:
                if yolo_lines:
                    f.write('\n'.join(yolo_lines))
                    count += 1
                else:
                    # Create empty file if image has no valid objects (important for background images)
                    pass 
                    empty_count += 1
            
        except Exception as e:
            print(f"   Error processing {xml_file.name}: {e}")

    print(f"   Created {count} label files (and {empty_count} empty files).")

def create_yaml():
    """Generates the data.yaml configuration file needed by YOLO."""
    yaml_path = DATA_ROOT / "data.yaml"
    
    data = {
        'path': str(DATA_ROOT.absolute()), # Absolute path to root
        'train': 'train/images',           # Relative path to images
        'val': 'validation/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(CLASSES)}
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
        
    print(f"\n📄 Configuration saved to: {yaml_path}")

if __name__ == "__main__":
    # Process all three splits
    process_folder("train")
    process_folder("validation")
    process_folder("test")
    
    # Generate the yaml config
    create_yaml()