import os
from ultralytics import YOLO

# ==========================================
# 1. SETUP PATHS
# ==========================================
dataset_path = "/dist_home/suryansh/sharukesh/analog/Datasets/Final_Blood_YOLO_Hierarchical_Remastered_clahe"
data_yaml = os.path.join(dataset_path, "data_balenced_pt2.yaml")
custom_model_yaml = "ultralytics/cfg/models/11/hemo-flash.yaml"

print(f"🚀  Initializing Hemo-Flash-v11 Training...")

# ==========================================
# 2. TRAIN CUSTOM ARCHITECTURE
# ==========================================
model = YOLO(custom_model_yaml) 

# Load pretrained weights
try:
    model.load("yolo11n.pt")
    print("✅ Loaded compatible weights from YOLOv11n")
except Exception as e:
    print(f"⚠️ Could not load pretrained weights: {e}")

print("🔥  Starting training with Valid Phase2 Imbalance Fixes...")

results = model.train(
    data=data_yaml,
    epochs=100,
    imgsz=640,
    batch=16,
    project=f"{dataset_path}/runs/train",
    name="hemo_flash_v11_9class_phase2_pt2_imb",
    plots=True,
    device=0,
    workers=8,
    
    # === PHASE 1: VALID IMBALANCE CONFIGURATION ===
    # 'cls' IS the valid argument to punish class confusion.
    # We increase it to 4.0 (Default 0.5) to make the model focus 
    # intensely on correctly identifying the rare cells.
    cls=1.5,        
    # === AUGMENTATION STRATEGIES ===
    mixup=0.15,          
    copy_paste=0.3,     
    mosaic=1.0,         
    
    # === GEOMETRIC AUGMENTATIONS ===
    degrees=180,    
    flipud=0.5,     
    fliplr=0.5,     
    scale=0.5,
    
    patience=0,    
    save=True,      
)

print("✅  Hemo-Flash Training Complete.")
