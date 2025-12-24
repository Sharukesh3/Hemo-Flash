from ultralytics import YOLO
from roboflow import Roboflow
import os

# ==========================================
# 1. SETUP DATASET
# ==========================================
print("⬇️  Checking for BCCD Dataset...")
dataset_location = "BCCD-2" 

if not os.path.exists(dataset_location):
    try:
        rf = Roboflow(api_key="PjUym3maiIS3wU8Ce7zp") # <--- PASTE KEY HERE
        project = rf.workspace("roboflow-100").project("bccd-ouzjz")
        dataset = project.version(2).download("yolov11")
        dataset_location = dataset.location
    except Exception as e:
        print("Assuming dataset is already in local folder 'BCCD-2'")

# ==========================================
# 2. TRAIN WITH PURE GEOMETRIC AUGMENTATION
# ==========================================
print("🚀 Starting Pure Augmented Baseline...")

model = YOLO("yolo11n.pt") 

results = model.train(
    data=f"{dataset_location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="baseline_aug_pure_yolo11n",
    project="hemo_tests",
    plots=True,
    
    # === GEOMETRIC AUGMENTATIONS ONLY ===
    # These teach the model shape/orientation invariance
    # without touching color or contrast.
    
    degrees=180,   # Rotation: +/- 180 degrees (Essential for cells)
    flipud=0.5,    # Flip Up-Down: 50% chance (Essential for microscopy)
    fliplr=0.5,    # Flip Left-Right: 50% chance
    
    scale=0.5,     # Scale: Zoom in/out by 50% (Helps with different magnifications)
    mosaic=1.0,    # Mosaic: Combines 4 images (Helps with small platelets)
    
    # === DISABLED/DEFAULT ===
    # We are NOT boosting HSV to mimic CLAHE. 
    # YOLO will use its mild defaults, but we aren't forcing heavy contrast changes.
)

print(f"✅ Training Complete. Check 'hemo_tests/baseline_aug_pure_yolo11n' for metrics.")
