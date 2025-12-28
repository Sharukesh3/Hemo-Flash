import os
from ultralytics import YOLO

# ==========================================
# 1. SETUP PATHS & CONFIG
# ==========================================
# Absolute path to your dataset config based on your pwd history
dataset_path = "/dist_home/suryansh/sharukesh/analog/Datasets/Final_Blood_YOLO_Hierarchical_Remastered_clahe"
data_yaml = os.path.join(dataset_path, "data.yaml")

print(f"🚀  Initializing Base Model Training...")
print(f"📂  Target Dataset: {data_yaml}")

# Verify data file exists before starting
if not os.path.exists(data_yaml):
    raise FileNotFoundError(f"❌ Could not find data.yaml at: {data_yaml}")

# ==========================================
# 2. TRAIN STANDARD BASELINE (YOLOv11n)
# ==========================================
# We load the official pre-trained weights to establish the "Control" performance
model = YOLO("yolo11n.pt") 

print("🔥  Starting training...")

results = model.train(
    data=data_yaml,
    epochs=100,             # Keeping consistency with your request
    imgsz=640,
    batch=16,               # Standard batch size for 11n
    
    # Organizing outputs clearly for comparison later
    project=f"{dataset_path}/runs/train", 
    name="baseline_yolo11n_hemo_new_data",
    
    # Key settings for benchmarking
    plots=True,             # Generate confusion matrices/curves
    exist_ok=True,          # Overwrite if exists (optional, distinct run names preferred usually)
    device=0,               # Force GPU (Assumes single GPU on compute node)
    workers=8               # Optimize dataloading
)

print("✅  Baseline Training Complete.")
print(f"📊  Results saved to: {dataset_path}/runs/train/baseline_yolo11n_hemo")
