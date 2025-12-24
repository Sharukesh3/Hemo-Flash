import os
from ultralytics import YOLO

# ==========================================
# 1. SETUP PATHS
# ==========================================
# Absolute path to your local dataset
dataset_path = "/dist_home/suryansh/sharukesh/analog/Datasets/Final_Blood_YOLO_Hierarchical"
data_yaml = os.path.join(dataset_path, "data.yaml")

print(f"🚀  Initializing Geometric Augmentation Training...")
print(f"📂  Target Dataset: {data_yaml}")

if not os.path.exists(data_yaml):
    raise FileNotFoundError(f"❌ Could not find data.yaml at: {data_yaml}")

# ==========================================
# 2. TRAIN WITH PURE GEOMETRIC AUGMENTATION
# ==========================================
# We start with the standard weights again to see how much augs help
model = YOLO("yolo11n.pt") 

print("🔥  Starting training with Heavy Geometric Augmentations...")

results = model.train(
    data=data_yaml,
    epochs=100,
    imgsz=640,
    batch=16,
    
    # Save runs inside your dataset folder for organization
    project=f"{dataset_path}/runs/train",
    name="baseline_aug_pure_yolo11n",
    
    plots=True,
    device=0,       # Force GPU usage
    workers=8,      # Efficient data loading
    
    # === GEOMETRIC AUGMENTATIONS ONLY ===
    # These teach the model shape/orientation invariance.
    # Highly effective for blood cells which appear at random angles.
    
    degrees=180,    # Rotation: +/- 180 degrees (Cells have no "up" or "down")
    flipud=0.5,     # Flip Up-Down: 50% chance (Microscope slides are 2D planes)
    fliplr=0.5,     # Flip Left-Right: 50% chance
    
    scale=0.5,      # Scale: Zoom in/out by 50% (Handles variations in cell size/zoom)
    mosaic=1.0,     # Mosaic: Combines 4 images (Crucial for dense object detection like platelets)
    
    # === DISABLED/DEFAULT ===
    # We are NOT adding color jitter (hsv_h, hsv_s, hsv_v) yet.
    # This isolates whether geometry alone improves detection.
)

print("✅  Augmented Training Complete.")
print(f"📊  Results saved to: {dataset_path}/runs/train/baseline_aug_pure_yolo11n")
