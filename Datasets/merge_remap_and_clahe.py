import os
import shutil
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
WORKSPACE = "Blood_Roboflow_Workspace_Remastered"  
OUTPUT_DIR = "Final_Blood_YOLO_Hierarchical_Remastered_clahe"
APPLY_CLAHE = True 

FINAL_CLASSES = [
    'RBC_Normal', 'RBC_Sickle', 'Platelets', 'WBC_Base', 
    'Neutrophil', 'Eosinophil', 'Basophil', 'Monocyte', 'Lymphocyte'
]

# --- CLAHE FUNCTION ---
def apply_clahe_to_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None: return None
    
    # Convert to LAB (L=Lightness)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge back
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final

def read_classes(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', [])

def get_mapping(dataset_name, old_names):
    mapping = {}
    print(f"\n🔍 Mapping classes for {dataset_name}: {old_names}")
    
    for i, name in enumerate(old_names):
        name_lower = name.lower()
        new_ids = []

        # --- 1. SICKLE LOGIC ---
        if any(x in name_lower for x in ['sickle', 'elongated', 'abnormal', 'vertical', 'submarine', 'ice cube']):
            new_ids = [1] 
            print(f"   ✅ {name} ({i}) -> RBC_Sickle (1)")

        # --- 2. PLATELET LOGIC ---
        elif 'platelet' in name_lower:
            new_ids = [2]

        # --- 3. WBC HIERARCHY ---
        elif 'neut' in name_lower:
            new_ids = [3, 4] 
        elif 'eosi' in name_lower:
            new_ids = [3, 5] 
        elif 'baso' in name_lower:
            new_ids = [3, 6] 
        elif 'mono' in name_lower:
            new_ids = [3, 7] 
        elif 'lymp' in name_lower:
            new_ids = [3, 8] 
            
        # --- 4. GENERIC WBC ---
        elif 'wbc' in name_lower or 'leuko' in name_lower:
            new_ids = [3] 
            print(f"   ⚠️  {name} ({i}) -> WBC_Base (3) ONLY")

        # --- 5. NORMAL RBC ---
        elif any(x in name_lower for x in ['rbc', 'red', 'circular', 'normal']):
            new_ids = [0] 

        if new_ids:
            mapping[i] = new_ids
        else:
            print(f"   ❌ IGNORING class: {name} ({i})")
            
    return mapping

# --- MAIN EXECUTION ---
print(f"🧹 Re-building output directory: {OUTPUT_DIR}...")
if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)

# Create Structure
for split in ['train', 'valid', 'test']:
    os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

datasets = ["BCCD", "Raabin_WBC", "Sickle_Cell"]

print(f"🎨 CLAHE Enhancement: {'ENABLED' if APPLY_CLAHE else 'DISABLED'}")

for ds_name in datasets:
    ds_path = Path(WORKSPACE) / ds_name
    yaml_path = ds_path / "data.yaml"
    if not yaml_path.exists(): continue
        
    old_classes = read_classes(yaml_path)
    remap_dict = get_mapping(ds_name, old_classes)
    
    for split in ['train', 'valid', 'test']:
        src_split_path = ds_path / split
        if not src_split_path.exists():
            if split == 'valid': src_split_path = ds_path / 'val'
        if not src_split_path.exists(): continue
        
        dst_split = split if split != 'val' else 'valid'
        images = list((src_split_path / "images").glob("*"))
        
        print(f"   📦 Processing {ds_name} [{split}]: {len(images)} images...")
        
        for img_path in tqdm(images, leave=False):
            # --- FIX: SKIP DIRECTORIES AND HIDDEN FILES ---
            if img_path.is_dir() or img_path.name.startswith('.'):
                continue
            
            new_filename = f"{ds_name}_{img_path.name}"
            target_img_path = f"{OUTPUT_DIR}/{dst_split}/images/{new_filename}"
            
            # --- APPLY CLAHE OR COPY ---
            if APPLY_CLAHE:
                processed_img = apply_clahe_to_image(img_path)
                if processed_img is not None:
                    cv2.imwrite(target_img_path, processed_img)
                else:
                    shutil.copy(img_path, target_img_path)
            else:
                shutil.copy(img_path, target_img_path)
            
            # --- HANDLE LABELS ---
            lbl_path = src_split_path / "labels" / f"{img_path.stem}.txt"
            if lbl_path.exists():
                new_lbl_path = f"{OUTPUT_DIR}/{dst_split}/labels/{ds_name}_{img_path.stem}.txt"
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts: continue
                    old_id = int(parts[0])
                    coords = ' '.join(parts[1:])
                    if old_id in remap_dict:
                        for new_id in remap_dict[old_id]:
                            new_lines.append(f"{new_id} {coords}")
                if new_lines:
                    with open(new_lbl_path, 'w') as f:
                        f.write("\n".join(new_lines))

# --- CREATE DATA.YAML ---
yaml_content = f"""
path: {os.path.abspath(OUTPUT_DIR)}
train: train/images
val: valid/images
test: test/images
nc: 9
names: {FINAL_CLASSES}
"""
with open(f"{OUTPUT_DIR}/data.yaml", 'w') as f:
    f.write(yaml_content)

print("\n🎉 DONE! Images merged, CLAHE applied, and classes remapped.")