import os
import random
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter

# --- CONFIGURATION ---
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
TRAIN_IMG_DIR = DATA_ROOT / "train" / "images"
VAL_IMG_DIR = DATA_ROOT / "validation" / "images"

# XML Folders (Source for reading old boxes, Destination for writing new combined boxes)
TRAIN_XML_DIR = DATA_ROOT / "train" / "annotations"
VAL_XML_DIR = DATA_ROOT / "validation" / "annotations"

TARGET_COUNTS = {
    "train": 40,
    "validation": 15
}

CATEGORIES = [
    'crazing', 'inclusion', 'patches',
    'pitted_surface', 'rolled-in_scale', 'scratches'
]

# --- XML HANDLING HELPERS ---

def get_original_boxes(image_filename, original_xml_dir):
    """
    Tries to find the original XML file and returns a list of existing bounding boxes.
    Returns: [(xmin, ymin, xmax, ymax), ...]
    """
    existing_boxes = []
    
    # Assumption: XML has same name as image but .xml extension
    xml_filename = os.path.splitext(image_filename)[0] + ".xml"
    xml_path = original_xml_dir / xml_filename
    
    if not xml_path.exists():
        # Fallback: Sometimes annotations are inside subfolders matching categories
        # We try to search recursively or check category subfolders if needed
        # For now, we return empty if not found immediately
        return []
        
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))
                existing_boxes.append((xmin, ymin, xmax, ymax))
                
    except Exception as e:
        print(f"Warning: Could not parse original XML {xml_path}: {e}")
        
    return existing_boxes

def create_xml_content(filename, folder_name, width, height, class_name, bboxes):
    """Generates the content string for a Pascal VOC XML file."""
    xml_content = [
        "<annotation>",
        f"    <folder>{folder_name}</folder>",
        f"    <filename>{filename}</filename>",
        "    <source>",
        "        <database>Synthetic_Augmentation</database>",
        "    </source>",
        "    <size>",
        f"        <width>{width}</width>",
        f"        <height>{height}</height>",
        "        <depth>3</depth>",
        "    </size>",
        "    <segmented>0</segmented>"
    ]
    
    for (xmin, ymin, xmax, ymax) in bboxes:
        # Clamp coordinates
        xmin = max(0, min(xmin, width - 1))
        ymin = max(0, min(ymin, height - 1))
        xmax = max(0, min(xmax, width - 1))
        ymax = max(0, min(ymax, height - 1))
        
        if xmin >= xmax or ymin >= ymax:
            continue
            
        xml_content.extend([
            "    <object>",
            f"        <name>{class_name}</name>",
            "        <pose>Unspecified</pose>",
            "        <truncated>0</truncated>",
            "        <difficult>0</difficult>",
            "        <bndbox>",
            f"            <xmin>{int(xmin)}</xmin>",
            f"            <ymin>{int(ymin)}</ymin>",
            f"            <xmax>{int(xmax)}</xmax>",
            f"            <ymax>{int(ymax)}</ymax>",
            "        </bndbox>",
            "    </object>"
        ])
        
    xml_content.append("</annotation>")
    return "\n".join(xml_content)

# --- DEFECT SIMULATION FUNCTIONS ---

def simulate_scratches(img):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    bboxes = []
    
    for _ in range(random.randint(3, 7)):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        length = random.randint(30, 100)
        angle = random.uniform(0, 2 * 3.14159)
        x2 = x1 + length * np.cos(angle)
        y2 = y1 + length * np.sin(angle)
        
        color = random.randint(150, 255)
        thickness = random.randint(1, 3)
        draw.line([(x1, y1), (x2, y2)], fill=(color, color, color), width=thickness)
        
        pad = 5
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        bboxes.append((xmin - pad, ymin - pad, xmax + pad, ymax + pad))
        
    return img, bboxes

def simulate_pitted_surface(img):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    bboxes = []

    for _ in range(random.randint(8, 15)):
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)
        c_xmin, c_ymin = center_x, center_y
        c_xmax, c_ymax = center_x, center_y

        for _ in range(random.randint(30, 70)):
            offset_x = random.randint(-40, 40)
            offset_y = random.randint(-40, 40)
            pt_x, pt_y = center_x + offset_x, center_y + offset_y
            
            c_xmin = min(c_xmin, pt_x)
            c_ymin = min(c_ymin, pt_y)
            c_xmax = max(c_xmax, pt_x)
            c_ymax = max(c_ymax, pt_y)

            color = random.randint(0, 30)
            radius = random.randint(1, 2)
            draw.ellipse([pt_x, pt_y, pt_x+radius, pt_y+radius], fill=(color, color, color))
            
        bboxes.append((c_xmin, c_ymin, c_xmax, c_ymax))
        
    return img, bboxes

def simulate_patches(img):
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    width, height = img.size
    bboxes = []

    for _ in range(random.randint(1, 3)):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(40, 100) 

        if random.random() > 0.5:
            color = (0, 0, 0, random.randint(50, 100)) 
        else:
            color = (200, 200, 200, random.randint(30, 80)) 

        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
        bboxes.append((x-r, y-r, x+r, y+r))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=15))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB"), bboxes

def simulate_inclusion(img):
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    width, height = img.size
    bboxes = []

    for _ in range(random.randint(1, 3)):
        center_x = random.randint(20, width-20)
        center_y = random.randint(50, height-100)
        c_xmin, c_ymin = center_x, center_y
        c_xmax, c_ymax = center_x, center_y

        num_streaks = random.randint(5, 15)
        for _ in range(num_streaks):
            offset_x = random.randint(-15, 15)
            offset_y = random.randint(-30, 30)
            start_x = center_x + offset_x
            start_y = center_y + offset_y
            length = random.randint(20, 60)
            end_x = start_x + random.randint(-2, 2)
            end_y = start_y + length

            c_xmin = min(c_xmin, start_x, end_x)
            c_ymin = min(c_ymin, start_y, end_y)
            c_xmax = max(c_xmax, start_x, end_x)
            c_ymax = max(c_ymax, start_y, end_y)

            thickness = random.randint(2, 5)
            alpha = random.randint(100, 180)
            color = (random.randint(20, 50), random.randint(20, 50), random.randint(20, 50), alpha)
            draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=thickness)
            
        bboxes.append((c_xmin, c_ymin, c_xmax, c_ymax))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB"), bboxes

def simulate_rolled_in_scale(img):
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    width, height = img.size
    bboxes = []

    for _ in range(random.randint(5, 10)):
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)
        patch_radius = random.randint(40, 100)
        
        bboxes.append((center_x - patch_radius, center_y - patch_radius, 
                       center_x + patch_radius, center_y + patch_radius))

        for _ in range(random.randint(50, 120)):
            offset_x = int(random.gauss(0, patch_radius/2.5))
            offset_y = int(random.gauss(0, patch_radius/2.5))
            flake_x = center_x + offset_x
            flake_y = center_y + offset_y
            
            flake_w = random.randint(3, 10)
            flake_h = random.randint(3, 8)
            color_val = random.randint(10, 50)
            alpha = random.randint(180, 240)
            color = (color_val, color_val, color_val, alpha)

            draw.ellipse(
                [flake_x - flake_w//2, flake_y - flake_h//2,
                 flake_x + flake_w//2, flake_y + flake_h//2],
                fill=color
            )

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.8))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB"), bboxes

def simulate_crazing(img):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    bboxes = []
    
    center_x = random.randint(50, width-50)
    center_y = random.randint(50, height-50)
    curr_x, curr_y = center_x, center_y
    min_x, min_y = curr_x, curr_y
    max_x, max_y = curr_x, curr_y
    
    for _ in range(random.randint(20, 40)):
        angle = random.uniform(0, 2 * 3.14159)
        length = random.randint(10, 30)
        
        next_x = curr_x + length * np.cos(angle)
        next_y = curr_y + length * np.sin(angle)
        
        draw.line([(curr_x, curr_y), (next_x, next_y)], fill=(20, 20, 20), width=1)
        
        min_x = min(min_x, next_x)
        min_y = min(min_y, next_y)
        max_x = max(max_x, next_x)
        max_y = max(max_y, next_y)
        curr_x, curr_y = next_x, next_y
        
    bboxes.append((min_x, min_y, max_x, max_y))
    return img, bboxes

def general_augmentations(img):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.9, 1.3))
    return img

DEFECT_MAP = {
    'crazing': simulate_crazing,
    'inclusion': simulate_inclusion,
    'patches': simulate_patches,
    'pitted_surface': simulate_pitted_surface,
    'rolled-in_scale': simulate_rolled_in_scale,
    'scratches': simulate_scratches
}

# --- MAIN LOOP ---

def process():
    print("Starting Synthetic Defect Generation with Combined (Old+New) Annotations...")

    for d in [TRAIN_XML_DIR, VAL_XML_DIR]:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            print(f"Created folder: {d}")

    for split, goal_count in TARGET_COUNTS.items():
        img_base_dir = TRAIN_IMG_DIR if split == "train" else VAL_IMG_DIR
        xml_base_dir = TRAIN_XML_DIR if split == "train" else VAL_XML_DIR
        
        print(f"\n=== Processing {split.upper()} set (Target: {goal_count} new per class) ===")

        for category in CATEGORIES:
            img_cat_path = img_base_dir / category
            
            if not img_cat_path.exists():
                print(f"Skipping {category} (path not found)")
                continue
                
            existing_files = [f for f in os.listdir(img_cat_path) if f.endswith(('.jpg', '.png')) and "aug_" not in f]
            
            if not existing_files:
                print(f"No original images found in {category}")
                continue

            selected_files = random.choices(existing_files, k=goal_count)
            print(f"   Generating {goal_count} augmented images for {category}...")
            
            for i, filename in enumerate(selected_files):
                try:
                    img_path = img_cat_path / filename
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        
                        # 1. READ OLD ANNOTATIONS
                        original_bboxes = get_original_boxes(filename, xml_base_dir)
                        
                        # 2. CREATE NEW DEFECTS
                        new_bboxes = []
                        if category in DEFECT_MAP:
                            img, new_bboxes = DEFECT_MAP[category](img)
                            
                        # 3. APPLY GENERAL AUGMENTATIONS
                        img = general_augmentations(img)
                        
                        # 4. MERGE BOXES
                        all_bboxes = original_bboxes + new_bboxes
                        
                        # Save Image
                        new_filename = f"aug_{split}_{category}_{i}_{int(random.random()*10000)}.jpg"
                        img.save(img_cat_path / new_filename, quality=95)
                        
                        # Save XML
                        xml_filename = new_filename.replace(".jpg", ".xml")
                        xml_content = create_xml_content(
                            new_filename, 
                            category, 
                            img.width, 
                            img.height, 
                            category, 
                            all_bboxes
                        )
                        
                        with open(xml_base_dir / xml_filename, "w") as f:
                            f.write(xml_content)
                        
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    print("\nDone! Images saved. Annotations merged (Old+New) and saved.")

if __name__ == "__main__":
    process()