import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
WORKSPACE = "Blood_Roboflow_Workspace"
OUTPUT_DIR = "Final_Blood_YOLO_Hierarchical"
FINAL_CLASSES = [
    'RBC_Normal', 'RBC_Sickle', 'Platelets', 'WBC_Base', 
    'Neutrophil', 'Eosinophil', 'Basophil', 'Monocyte', 'Lymphocyte'
]

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

        # --- 1. SICKLE LOGIC (The Fix) ---
        # We now catch "vertical", "submarine", "ice cube" as Sickle Cells
        if any(x in name_lower for x in ['sickle', 'elongated', 'abnormal', 'vertical', 'submarine', 'ice cube']):
            new_ids = [1] # RBC_Sickle
            print(f"   ✅ {name} ({i}) -> RBC_Sickle (1)")

        # --- 2. PLATELET LOGIC ---
        elif 'platelet' in name_lower:
            new_ids = [2] # Platelets

        # --- 3. WBC HIERARCHY ---
        elif 'neut' in name_lower:
            new_ids = [3, 4] # WBC_Base + Neutrophil
        elif 'eosi' in name_lower:
            new_ids = [3, 5] # WBC_Base + Eosinophil
        elif 'baso' in name_lower:
            new_ids = [3, 6] # WBC_Base + Basophil
        elif 'mono' in name_lower:
            new_ids = [3, 7] # WBC_Base + Monocyte
        elif 'lymp' in name_lower:
            new_ids = [3, 8] # WBC_Base + Lymphocyte
            
        # --- 4. GENERIC WBC ---
        elif 'wbc' in name_lower or 'leuko' in name_lower:
            new_ids = [3] # WBC_Base ONLY
            print(f"   ⚠️  {name} ({i}) -> WBC_Base (3) ONLY")

        # --- 5. NORMAL RBC ---
        elif any(x in name_lower for x in ['rbc', 'red', 'circular', 'normal']):
            new_ids = [0] # RBC_Normal

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
            new_filename = f"{ds_name}_{img_path.name}"
            shutil.copy(img_path, f"{OUTPUT_DIR}/{dst_split}/images/{new_filename}")
            
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

print("\n🎉 FIXED! 'Submarine' & 'Vertical' are now correctly labeled as Sickle Cells.")
