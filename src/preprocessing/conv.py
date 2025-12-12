import os
import xml.etree.ElementTree as ET
from pathlib import Path

# --- CONFIGURATION ---
# Define the root path relative to this script location
# Path: project_root/data/raw
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

# Class names must match the order in your data.yaml exactly
CLASSES = [
    'crazing',
    'inclusion',
    'patches',
    'pitted_surface',
    'rolled-in_scale',
    'scratches'
]


def convert_box(size, box):
    """
    Converts XML (xmin, xmax, ymin, ymax) to YOLO (x_center, y_center, width, height).
    All values are normalized between 0 and 1.
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    # Calculate center, width, and height
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]

    # Normalize
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_folder(split_name):
    """
    Converts all XML files in a specific split (train/validation) to TXT.
    """
    # Define paths
    img_dir = DATA_ROOT / split_name / "images"
    xml_dir = DATA_ROOT / split_name / "annotations"
    label_dir = DATA_ROOT / split_name / "labels"

    # Create the labels folder if it doesn't exist
    if not label_dir.exists():
        label_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {label_dir}")

    if not xml_dir.exists():
        print(f"Skipping {split_name}: Annotations folder not found at {xml_dir}")
        return

    print(f"Processing {split_name} set...")

    converted_count = 0

    # Iterate over every XML file
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue

        try:
            tree = ET.parse(xml_dir / xml_file)
            root = tree.getroot()

            # Get image dimensions
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            # Prepare output text file path
            txt_filename = xml_file.replace('.xml', '.txt')
            txt_path = label_dir / txt_filename

            with open(txt_path, 'w') as out_file:
                for obj in root.iter('object'):
                    cls_name = obj.find('name').text

                    if cls_name not in CLASSES:
                        print(f"Warning: Class '{cls_name}' not in list, skipping object in {xml_file}")
                        continue

                    cls_id = CLASSES.index(cls_name)

                    # Get bounding box
                    xmlbox = obj.find('bndbox')
                    b = (
                        float(xmlbox.find('xmin').text),
                        float(xmlbox.find('xmax').text),
                        float(xmlbox.find('ymin').text),
                        float(xmlbox.find('ymax').text)
                    )

                    # Convert to YOLO format
                    bb = convert_box((w, h), b)

                    # Write line: class_id x_center y_center width height
                    out_file.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

            converted_count += 1

        except Exception as e:
            print(f"Error converting {xml_file}: {e}")

    print(f"Converted {converted_count} files in {split_name}.")


if __name__ == "__main__":
    print("Starting XML to YOLO TXT conversion...")
    convert_folder("train")
    convert_folder("validation")
    print("\nConversion Complete! You can now run the training script.")