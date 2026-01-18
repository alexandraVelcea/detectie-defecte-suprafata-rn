import os
import random
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter

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

# Counts
MOVE_FROM_TRAIN_TO_POOL = 30   # Move 30 from Train to "Pool"
TARGET_VAL_COUNT        = 45   # Final Count for Validation
TARGET_TEST_COUNT       = 45   # Final Count for Test
AUGMENT_IN_TRAIN        = 105  # Images to process in Train

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

# --- MAIN LOGIC ---

def move_file_pair(filename, src_img_dir, src_xml_dir, dest_img_dir, dest_xml_dir):
    """Safely moves an image and its XML."""
    if not (src_img_dir / filename).exists():
        return # Skip if source missing

    # Move Image
    shutil.move(str(src_img_dir / filename), str(dest_img_dir / filename))
    
    # Move XML
    xml_name = filename.replace(".jpg", ".xml").replace(".png", ".xml")
    if (src_xml_dir / xml_name).exists():
        shutil.move(str(src_xml_dir / xml_name), str(dest_xml_dir / xml_name))

def process_category(category):
    # Current Raw Paths
    train_cat_img_dir = RAW_TRAIN_IMG_DIR / category
    
    # Validation Pool Logic
    # (Assuming raw/validation is either flat or categorized, we handle categorized for input)
    if (RAW_VAL_IMG_DIR / category).exists():
        val_pool_src = RAW_VAL_IMG_DIR / category
    else:
        val_pool_src = RAW_VAL_IMG_DIR

    if not train_cat_img_dir.exists(): 
        return

    # --- STEP 1: MOVE 30 FROM TRAIN TO VAL POOL ---
    # Identify clean originals
    originals = [f for f in os.listdir(train_cat_img_dir) if f.lower().endswith(('.jpg', '.png')) and "aug_" not in f]
    random.shuffle(originals)
    
    to_move = originals[:MOVE_FROM_TRAIN_TO_POOL]
    print(f"[{category}] Step 1: Moving {len(to_move)} images from Train -> Raw Validation Pool.")
    
    # Staging area for pool (temp use raw/validation/category)
    (RAW_VAL_IMG_DIR / category).mkdir(parents=True, exist_ok=True)
    
    for f in to_move:
        move_file_pair(f, train_cat_img_dir, RAW_TRAIN_XML_DIR, RAW_VAL_IMG_DIR / category, RAW_VAL_XML_DIR)

    # --- STEP 2: SPLIT TOTAL POOL (45 Val / 45 Test) ---
    pool_dir = RAW_VAL_IMG_DIR / category
    pool_files = [f for f in os.listdir(pool_dir) if f.lower().endswith(('.jpg', '.png'))]
    random.shuffle(pool_files)
    
    if len(pool_files) < (TARGET_VAL_COUNT + TARGET_TEST_COUNT):
        print(f"Warning: Total pool for {category} is {len(pool_files)}. Splitting 50/50.")
        val_set = pool_files[:len(pool_files)//2]
        test_set = pool_files[len(pool_files)//2:]
    else:
        val_set = pool_files[:TARGET_VAL_COUNT]
        test_set = pool_files[TARGET_VAL_COUNT : TARGET_VAL_COUNT + TARGET_TEST_COUNT]

    print(f"[{category}] Step 2: Splitting Pool ({len(pool_files)}) -> {len(val_set)} Validation, {len(test_set)} Test.")

    for f in val_set:
        move_file_pair(f, pool_dir, RAW_VAL_XML_DIR, FINAL_VAL_IMG_DIR, FINAL_VAL_XML_DIR)
        
    for f in test_set:
        move_file_pair(f, pool_dir, RAW_VAL_XML_DIR, FINAL_TEST_IMG_DIR, FINAL_TEST_XML_DIR)

    # --- STEP 3: AUGMENT REMAINING TRAIN ---
    remaining_train = [f for f in os.listdir(train_cat_img_dir) if f.lower().endswith(('.jpg', '.png')) and "aug_" not in f]
    to_augment = remaining_train[:AUGMENT_IN_TRAIN]
    
    print(f"[{category}] Step 3: Augmenting {len(to_augment)} images in Train.")

    for i, filename in enumerate(to_augment):
        try:
            file_path = train_cat_img_dir / filename
            xml_path = RAW_TRAIN_XML_DIR / filename.replace(".jpg", ".xml")
            
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                old_boxes = get_boxes_from_xml(xml_path)
                
                defect_func = DEFECT_MAP.get(category, simulate_scratches)
                img_def, new_boxes = defect_func(img)
                final_img = general_augmentations(img_def)
                
                aug_name = f"aug_{category}_{i}_{random.randint(100,999)}.jpg"
                final_img.save(train_cat_img_dir / aug_name)
                
                all_boxes = old_boxes + new_boxes
                aug_xml_content = create_xml_content(aug_name, final_img.width, final_img.height, category, all_boxes)
                with open(RAW_TRAIN_XML_DIR / aug_name.replace(".jpg", ".xml"), "w") as f:
                    f.write(aug_xml_content)
            
            # Delete Original
            os.remove(file_path)
            if xml_path.exists():
                os.remove(xml_path)

        except Exception as e:
            print(f"Error augmenting {filename}: {e}")

    # --- STEP 4: FLATTEN TO FINAL TRAIN FOLDER ---
    # Move all files (which are now only augmented/valid training data) from raw/train/cat -> final/train
    print(f"[{category}] Step 4: Moving & Flattening to {FINAL_TRAIN_IMG_DIR}")
    
    # Re-scan for all images in this category folder (should be mostly augmented ones now)
    all_final_files = [f for f in os.listdir(train_cat_img_dir) if f.lower().endswith(('.jpg', '.png'))]
    
    for f in all_final_files:
        move_file_pair(f, train_cat_img_dir, RAW_TRAIN_XML_DIR, FINAL_TRAIN_IMG_DIR, FINAL_TRAIN_XML_DIR)

    # Optional: Remove empty category folder in raw/train
    try:
        os.rmdir(train_cat_img_dir)
    except: pass

def main():
    # 1. Create Final Output Directories
    for d in [FINAL_TRAIN_IMG_DIR, FINAL_TRAIN_XML_DIR, 
              FINAL_VAL_IMG_DIR, FINAL_VAL_XML_DIR, 
              FINAL_TEST_IMG_DIR, FINAL_TEST_XML_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Ensure intermediate staging exists
    RAW_VAL_XML_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Process Each Category
    for cat in CATEGORIES:
        process_category(cat)

    # 3. Cleanup Raw Train Root if empty
    try:
        if RAW_TRAIN_IMG_DIR.exists() and not any(RAW_TRAIN_IMG_DIR.iterdir()):
             os.rmdir(RAW_TRAIN_IMG_DIR)
    except: pass

    print("\nProcess Complete!")
    print(f"Train (Flattened): {FINAL_TRAIN_IMG_DIR}")
    print(f"Validation:        {FINAL_VAL_IMG_DIR}")
    print(f"Test:              {FINAL_TEST_IMG_DIR}")

if __name__ == "__main__":
    main()