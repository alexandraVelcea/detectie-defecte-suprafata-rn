import os
import random
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter

# ---------- COD AUGMENTARE, DISTRIBUIRE DATE ----------

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Input Paths (Raw Data)
RAW_TRAIN_IMG_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "images"
RAW_TRAIN_XML_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "annotations"
RAW_VAL_IMG_DIR   = PROJECT_ROOT / "data" / "raw" / "validation" / "images"
RAW_VAL_XML_DIR   = PROJECT_ROOT / "data" / "raw" / "validation" / "annotations"

# Final Output Paths
FINAL_TRAIN_IMG_DIR = PROJECT_ROOT / "data" / "train" / "images"
FINAL_TRAIN_XML_DIR = PROJECT_ROOT / "data" / "train" / "annotations"
FINAL_VAL_IMG_DIR   = PROJECT_ROOT / "data" / "validation" / "images"
FINAL_VAL_XML_DIR   = PROJECT_ROOT / "data" / "validation" / "annotations"
FINAL_TEST_IMG_DIR  = PROJECT_ROOT / "data" / "test" / "images"
FINAL_TEST_XML_DIR  = PROJECT_ROOT / "data" / "test" / "annotations"

# Targets
TARGET_VAL_ORIGINALS  = 30   # Base number of REAL images for Validation
TARGET_TEST_ORIGINALS = 30   # Base number of REAL images for Test
# Train gets the rest

# TARGET RATIO: 40% of the FINAL dataset must be augmented
AUG_TARGET_RATIO = 0.40 

CATEGORIES = [
    'crazing', 'inclusion', 'patches', 
    'pitted_surface', 'rolled-in_scale', 'scratches'
]

# --- XML UTILS ---
def get_boxes_from_xml(xml_path):
    boxes = []
    if xml_path.exists():
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                if bbox is not None:
                    boxes.append((
                        int(float(bbox.find('xmin').text)),
                        int(float(bbox.find('ymin').text)),
                        int(float(bbox.find('xmax').text)),
                        int(float(bbox.find('ymax').text))
                    ))
        except: pass
    return boxes

def create_xml_content(filename, width, height, class_name, bboxes):
    xml = [
        "<annotation>",
        f"    <filename>{filename}</filename>",
        f"    <size><width>{width}</width><height>{height}</height><depth>3</depth></size>",
        "    <segmented>0</segmented>"
    ]
    for (xmin, ymin, xmax, ymax) in bboxes:
        xmin, xmax = max(0, xmin), min(width, xmax)
        ymin, ymax = max(0, ymin), min(height, ymax)
        xml.append(f"    <object><name>{class_name}</name><bndbox><xmin>{int(xmin)}</xmin><ymin>{int(ymin)}</ymin><xmax>{int(xmax)}</xmax><ymax>{int(ymax)}</ymax></bndbox></object>")
    xml.append("</annotation>")
    return "\n".join(xml)

# --- DEFECT SIMULATORS ---
def general_augmentations(img):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.9, 1.3))
    return img

def simulate_scratches(img):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    bboxes = []
    for _ in range(random.randint(3, 7)):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        length = random.randint(30, 100)
        angle = random.uniform(0, 2 * np.pi)
        x2 = x1 + length * np.cos(angle)
        y2 = y1 + length * np.sin(angle)
        color = random.randint(150, 255)
        draw.line([(x1, y1), (x2, y2)], fill=(color, color, color), width=random.randint(1, 3))
        pad = 5
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        bboxes.append((xmin - pad, ymin - pad, xmax + pad, ymax + pad))
    return img, bboxes

def simulate_pitted_surface(img):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    bboxes = []
    for _ in range(random.randint(5, 12)):
        cx, cy = random.randint(0, width), random.randint(0, height)
        c_xmin, c_ymin, c_xmax, c_ymax = cx, cy, cx, cy
        for _ in range(random.randint(20, 50)):
            ox, oy = random.randint(-30, 30), random.randint(-30, 30)
            px, py = cx + ox, cy + oy
            c_xmin, c_ymin = min(c_xmin, px), min(c_ymin, py)
            c_xmax, c_ymax = max(c_xmax, px), max(c_ymax, py)
            color = random.randint(0, 50)
            draw.ellipse([px, py, px+random.randint(1,2), py+random.randint(1,2)], fill=(color, color, color))
        bboxes.append((c_xmin, c_ymin, c_xmax, c_ymax))
    return img, bboxes

def simulate_patches(img):
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    width, height = img.size
    bboxes = []
    for _ in range(random.randint(1, 3)):
        x, y = random.randint(0, width), random.randint(0, height)
        r = random.randint(30, 80)
        color = (0, 0, 0, random.randint(50, 100)) if random.random() > 0.5 else (200, 200, 200, random.randint(30, 80))
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
        bboxes.append((x-r, y-r, x+r, y+r))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=10))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB"), bboxes

def simulate_inclusion(img):
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    width, height = img.size
    bboxes = []
    for _ in range(random.randint(1, 3)):
        sx, sy = random.randint(20, width-20), random.randint(20, height-20)
        length = random.randint(15, 50)
        ex, ey = sx + random.randint(-5, 5), sy + length
        draw.line([(sx, sy), (ex, ey)], fill=(30, 30, 30, random.randint(100, 180)), width=random.randint(2, 4))
        bboxes.append((min(sx, ex)-2, min(sy, ey)-2, max(sx, ex)+2, max(sy, ey)+2))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB"), bboxes

def simulate_rolled_in_scale(img):
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    width, height = img.size
    bboxes = []
    for _ in range(random.randint(3, 6)):
        cx, cy = random.randint(0, width), random.randint(0, height)
        r = random.randint(20, 60)
        bboxes.append((cx - r, cy - r, cx + r, cy + r))
        for _ in range(random.randint(30, 80)):
            ox, oy = int(random.gauss(0, r/3)), int(random.gauss(0, r/3))
            val = random.randint(20, 60)
            draw.ellipse([cx+ox, cy+oy, cx+ox+5, cy+oy+5], fill=(val, val, val, random.randint(150, 220)))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB"), bboxes

def simulate_crazing(img):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    bboxes = []
    cx, cy = random.randint(50, width-50), random.randint(50, height-50)
    curr_x, curr_y = cx, cy
    min_x, min_y, max_x, max_y = cx, cy, cx, cy
    for _ in range(random.randint(15, 30)):
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(5, 20)
        nx, ny = curr_x + length * np.cos(angle), curr_y + length * np.sin(angle)
        draw.line([(curr_x, curr_y), (nx, ny)], fill=(40, 40, 40), width=1)
        min_x, max_x = min(min_x, nx), max(max_x, nx)
        min_y, max_y = min(min_y, ny), max(max_y, ny)
        curr_x, curr_y = nx, ny
    bboxes.append((min_x, min_y, max_x, max_y))
    return img, bboxes

DEFECT_MAP = {
    'crazing': simulate_crazing,
    'inclusion': simulate_inclusion,
    'patches': simulate_patches,
    'pitted_surface': simulate_pitted_surface,
    'rolled-in_scale': simulate_rolled_in_scale,
    'scratches': simulate_scratches
}

# --- LOGIC HELPERS ---

def copy_file_pair(filename, src_img_dir, src_xml_dir, dest_img_dir, dest_xml_dir):
    """Copies file + XML to destination."""
    if not (src_img_dir / filename).exists():
        return
    shutil.copy(str(src_img_dir / filename), str(dest_img_dir / filename))
    
    xml_name = filename.replace(".jpg", ".xml").replace(".png", ".xml")
    if (src_xml_dir / xml_name).exists():
        shutil.copy(str(src_xml_dir / xml_name), str(dest_xml_dir / xml_name))

def augment_dataset(source_file_list, dest_img_dir, dest_xml_dir, category):
    """
    Generates synthetic images based on the source list provided.
    Ensures final ratio of Augmented data is 40%.
    """
    num_originals = len(source_file_list)
    if num_originals == 0:
        return

    # Calculate how many to generate
    # Target: Aug = 40% of Total (Orig + Aug)
    # Orig = 60% of Total -> Total = Orig / 0.6
    final_total_target = int(num_originals / (1.0 - AUG_TARGET_RATIO))
    num_to_augment = final_total_target - num_originals
    
    print(f"   -> Originals: {num_originals} | Generating {num_to_augment} augmented images.")

    for i in range(num_to_augment):
        try:
            # 1. Pick random original from the list
            original_data = random.choice(source_file_list)
            filename, _, _ = original_data
            
            # Read from destination (since we copied originals there already)
            src_img_path = dest_img_dir / filename
            src_xml_path = dest_xml_dir / filename.replace(".jpg", ".xml").replace(".png", ".xml")
            
            with Image.open(src_img_path) as img:
                img = img.convert("RGB")
                old_boxes = get_boxes_from_xml(src_xml_path)
                
                # 2. Simulate Defect
                defect_func = DEFECT_MAP.get(category, simulate_scratches)
                img_def, new_boxes = defect_func(img)
                
                # 3. General Augmentation
                final_img = general_augmentations(img_def)
                
                # 4. Save
                aug_name = f"aug_{category}_{i}_{random.randint(1000,9999)}.jpg"
                final_img.save(dest_img_dir / aug_name)
                
                all_boxes = old_boxes + new_boxes
                aug_xml_content = create_xml_content(aug_name, final_img.width, final_img.height, category, all_boxes)
                with open(dest_xml_dir / aug_name.replace(".jpg", ".xml"), "w") as f:
                    f.write(aug_xml_content)
                    
        except Exception as e:
            print(f"Error augmenting: {e}")

def process_category(category):
    print(f"\n--- Processing: {category} ---")
    
    # 1. Gather all available raw images (Pool)
    raw_pool = []
    if (RAW_TRAIN_IMG_DIR / category).exists():
        raw_pool.extend([(f, RAW_TRAIN_IMG_DIR / category, RAW_TRAIN_XML_DIR) 
                         for f in os.listdir(RAW_TRAIN_IMG_DIR / category) if f.lower().endswith(('.jpg', '.png'))])
    if (RAW_VAL_IMG_DIR / category).exists():
        raw_pool.extend([(f, RAW_VAL_IMG_DIR / category, RAW_VAL_XML_DIR) 
                         for f in os.listdir(RAW_VAL_IMG_DIR / category) if f.lower().endswith(('.jpg', '.png'))])
    
    if not raw_pool:
        print(f"Skipping {category}: No images found.")
        return

    random.shuffle(raw_pool)
    print(f"Total Raw Images found: {len(raw_pool)}")

    # 2. Split Data (Originals)
    if len(raw_pool) < (TARGET_VAL_ORIGINALS + TARGET_TEST_ORIGINALS):
        n_test = int(len(raw_pool) * 0.2)
        n_val = int(len(raw_pool) * 0.2)
    else:
        n_test = TARGET_TEST_ORIGINALS
        n_val = TARGET_VAL_ORIGINALS

    test_set = raw_pool[:n_test]
    val_set = raw_pool[n_test : n_test + n_val]
    train_set = raw_pool[n_test + n_val:]

    print(f"Originals Split -> Test: {len(test_set)} | Val: {len(val_set)} | Train: {len(train_set)}")

    # 3. Copy Originals
    for f_name, src_dir, src_xml_dir in test_set:
        copy_file_pair(f_name, src_dir, src_xml_dir, FINAL_TEST_IMG_DIR, FINAL_TEST_XML_DIR)
    for f_name, src_dir, src_xml_dir in val_set:
        copy_file_pair(f_name, src_dir, src_xml_dir, FINAL_VAL_IMG_DIR, FINAL_VAL_XML_DIR)
    for f_name, src_dir, src_xml_dir in train_set:
        copy_file_pair(f_name, src_dir, src_xml_dir, FINAL_TRAIN_IMG_DIR, FINAL_TRAIN_XML_DIR)

    # 4. Augment ALL sets (Test, Val, Train) to reach 40%
    print(f"Augmenting TEST Set...")
    augment_dataset(test_set, FINAL_TEST_IMG_DIR, FINAL_TEST_XML_DIR, category)
    
    print(f"Augmenting VAL Set...")
    augment_dataset(val_set, FINAL_VAL_IMG_DIR, FINAL_VAL_XML_DIR, category)
    
    print(f"Augmenting TRAIN Set...")
    augment_dataset(train_set, FINAL_TRAIN_IMG_DIR, FINAL_TRAIN_XML_DIR, category)

def main():
    # Create Dirs
    for d in [FINAL_TRAIN_IMG_DIR, FINAL_TRAIN_XML_DIR, 
              FINAL_VAL_IMG_DIR, FINAL_VAL_XML_DIR, 
              FINAL_TEST_IMG_DIR, FINAL_TEST_XML_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    print("--- STARTING DATASET GENERATION (Global 40% Augmentation) ---")

    for cat in CATEGORIES:
        process_category(cat)

    print("\n✅ Process Complete!")
    print(f"Train: {FINAL_TRAIN_IMG_DIR}")
    print(f"Val:   {FINAL_VAL_IMG_DIR}")
    print(f"Test:  {FINAL_TEST_IMG_DIR}")

if __name__ == "__main__":
    main()