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
    """Converts Pascal VOC (xmin, ymin, xmax, ymax) to YOLO (x_center, y_center, w, h)."""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def process_folder(split_name):
    """
    Converts XMLs in data/{split_name}/annotations to TXTs in data/{split_name}/labels
    """
    img_dir = DATA_ROOT / split_name / "images"
    xml_dir = DATA_ROOT / split_name / "annotations"
    label_dir = DATA_ROOT / split_name / "labels"
    
    if not xml_dir.exists():
        print(f"Skipping {split_name} (no annotations found)")
        return

    # Create labels folder
    label_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {split_name} data...")
    count = 0
    
    for xml_file in xml_dir.glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get Image Size
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            # Prepare YOLO lines
            yolo_lines = []
            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                if cls_name not in CLASSES:
                    continue
                
                cls_id = CLASSES.index(cls_name)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                
                bb = convert_box((w, h), b)
                yolo_lines.append(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")

            # Write TXT file
            txt_filename = xml_file.stem + ".txt"
            with open(label_dir / txt_filename, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            count += 1
            
        except Exception as e:
            print(f"Error processing {xml_file.name}: {e}")

    print(f"Created {count} label files for {split_name}.")

def create_yaml():
    """Generates the data.yaml configuration file."""
    yaml_path = DATA_ROOT / "data.yaml"
    
    # YOLO expects paths relative to the execution directory or absolute paths
    # We use absolute paths to be safe
    data = {
        'path': str(DATA_ROOT.absolute()),
        'train': 'train/images',
        'val': 'validation/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(CLASSES)}
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
        
    print(f"\nConfiguration saved to: {yaml_path}")

if __name__ == "__main__":
    process_folder("train")
    process_folder("validation")
    process_folder("test")
    create_yaml()