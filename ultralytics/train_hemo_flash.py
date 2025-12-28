import os
from ultralytics import YOLO

# ==========================================
# 1. SETUP PATHS
# ==========================================
# Absolute path to your local dataset
dataset_path = "/dist_home/suryansh/sharukesh/analog/Datasets/Final_Blood_YOLO_Hierarchical_Remastered_clahe"
data_yaml = os.path.join(dataset_path, "data.yaml")

# Path to your custom architecture YAML file
custom_model_yaml = "ultralytics/cfg/models/11/hemo-flash.yaml"

print(f"🚀  Initializing Hemo-Flash-v11 Training...")
print(f"📂  Target Dataset: {data_yaml}")
print(f"🏗️  Model Architecture: {custom_model_yaml}")

# Verify paths exist before crashing later
if not os.path.exists(data_yaml):
    raise FileNotFoundError(f"❌ Could not find data.yaml at: {data_yaml}")

if not os.path.exists(custom_model_yaml):
    raise FileNotFoundError(f"❌ Could not find hemo-flash.yaml at: {custom_model_yaml}")

# ==========================================
# 2. TRAIN CUSTOM ARCHITECTURE
# ==========================================
# Load the CUSTOM architecture (Builds from scratch based on YAML)
model = YOLO(custom_model_yaml) 

# OPTIONAL: Load pretrained weights into the compatible layers (Backbone)
# This helps convergence significantly. 
# It will warn about missing P5 layers, which is expected/good.
try:
    model.load("yolo11n.pt")
    print("✅ Loaded compatible weights from YOLOv11n")
except Exception as e:
    print(f"⚠️ Could not load pretrained weights: {e}")

print("🔥  Starting training with Heavy Geometric Augmentations...")

results = model.train(
    data=data_yaml,
    epochs=100,
    imgsz=640,
    batch=16,
    
    # Save runs inside your dataset folder for organization
    project=f"{dataset_path}/runs/train",
    name="hemo_flash_v11_9class_aug",
    
    plots=True,
    device=0,       # Force GPU usage
    workers=8,      # Efficient data loading
    
    # === GEOMETRIC AUGMENTATIONS ONLY ===
    # These teach the model shape/orientation invariance.
    degrees=180,    # Rotation: +/- 180 degrees
    flipud=0.5,     # Flip Up-Down: 50% chance
    fliplr=0.5,     # Flip Left-Right: 50% chance
    
    scale=0.5,      # Scale: Zoom in/out by 50%
    mosaic=1.0,     # Mosaic: Combines 4 images
    
    # === OPTIMIZATION ===
    patience=15,    # Stop if no improvement for 15 epochs
    save=True,      # Save checkpoints
)

print("✅  Hemo-Flash Training Complete.")
print(f"📊  Results saved to: {dataset_path}/runs/train/hemo_flash_v11_9class_aug")