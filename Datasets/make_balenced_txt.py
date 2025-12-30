import os
from pathlib import Path
from collections import Counter
import yaml
from tqdm import tqdm

# ================= CONFIGURATION =================
# 1. Path to your dataset's data.yaml
DATA_YAML_PATH = "/dist_home/suryansh/sharukesh/analog/Datasets/Final_Blood_YOLO_Hierarchical_Remastered_clahe/data.yaml"

# 2. Define which classes are "Rare" and how much to boost them.
# Based on your image: RBC_Sickle(1), Neutrophil(4), Eosinophil(5), 
# Basophil(6), Monocyte(7), Lymphocyte(8) are the rare ones.
# We will boost images containing these classes by 20x.
OVERSAMPLE_CONFIG = {
    1: 100,  # RBC_Sickle
    4: 10,  # Neutrophil
    5: 10,  # Eosinophil
    6: 10,  # Basophil
    7: 10,  # Monocyte
    8: 10   # Lymphocyte
}
# =================================================

def create_balanced_dataset():
    # Load data.yaml to get paths
    with open(DATA_YAML_PATH, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    # Resolve paths (Handle relative paths in yaml)
    base_path = Path(DATA_YAML_PATH).parent
    train_images_dir = base_path / data_cfg['train']
    
    # If the yaml path points to a .txt file, read it. If directory, scan it.
    if str(data_cfg['train']).endswith('.txt'):
        with open(base_path / data_cfg['train'], 'r') as f:
            image_files = [line.strip() for line in f.readlines()]
            image_files = [Path(p) if os.path.isabs(p) else base_path / p for p in image_files]
    else:
        # Assuming images are in a folder
        image_files = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png"))

    print(f"🧐 Scanning {len(image_files)} images for rare classes...")

    # We need to find the corresponding label for each image
    # Assuming standard YOLO layout: images/train/x.jpg -> labels/train/x.txt
    new_train_list = []
    stats = Counter()

    for img_path in tqdm(image_files):
        img_path = Path(img_path)
        
        # Construct label path
        # Replace 'images' dir with 'labels' and extension with .txt
        parts = list(img_path.parts)
        try:
            # Simple replacement strategy (works for standard structure)
            idx = parts.index("images")
            parts[idx] = "labels"
        except ValueError:
            # Fallback if structure is weird
            print(f"⚠️ specific structure issue with {img_path}, skipping...")
            continue
            
        label_path = Path(*parts).with_suffix(".txt")
        
        if not label_path.exists():
            # If no label, it's a background image (keep it once)
            new_train_list.append(str(img_path))
            continue

        # Read labels to see if a rare class is present
        has_rare = False
        max_multiplier = 1
        
        with open(label_path, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                stats[class_id] += 1
                
                # Check if this class is in our boost list
                if class_id in OVERSAMPLE_CONFIG:
                    has_rare = True
                    # Use the highest multiplier if an image has multiple rare classes
                    max_multiplier = max(max_multiplier, OVERSAMPLE_CONFIG[class_id])

        # Add to list
        if has_rare:
            # DUPLICATE THE PATH multiple times
            for _ in range(max_multiplier):
                new_train_list.append(str(img_path))
        else:
            # Majority class image, add once
            new_train_list.append(str(img_path))

    # Write the new text file
    output_txt = base_path / "train_balanced_pt2.txt"
    with open(output_txt, 'w') as f:
        for line in new_train_list:
            f.write(f"{line}\n")

    print(f"\n✅ Created Balanced Manifest: {output_txt}")
    print(f"📊 Original Count: {len(image_files)}")
    print(f"🚀 Balanced Count: {len(new_train_list)} (Effective Weighted Loader)")
    print("\n--- Class Counts (Raw Instances) ---")
    for cls_id, count in sorted(stats.items()):
        print(f"Class {cls_id}: {count}")

    print("\n⚠️  ACTION REQUIRED: Update your data.yaml to point 'train:' to this new .txt file!")

if __name__ == "__main__":
    create_balanced_dataset()
