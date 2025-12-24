from ultralytics import YOLO
from roboflow import Roboflow

# ==========================================
# 1. PREPARE DATASET
# ==========================================
# (We reuse the download logic so it's consistent)
print("⬇️  Checking for BCCD Dataset...")
try:
    rf = Roboflow(api_key="PjUym3maiIS3wU8Ce7zp") # <--- PASTE YOUR KEY HERE
    project = rf.workspace("roboflow-100").project("bccd-ouzjz")
    dataset = project.version(2).download("yolov11")
except Exception as e:
    print(f"⚠️  Note: If dataset is already downloaded, we will use the local copy.")
    # Assuming standard folder name if download was skipped/failed
    # You might need to adjust this path if you downloaded it manually
    dataset_location = "BCCD-2" 
else:
    dataset_location = dataset.location

# ==========================================
# 2. TRAIN STANDARD BASELINE
# ==========================================
print("🚀 Starting Standard YOLOv11n Baseline Training...")

# Load the OFFICIAL standard model
# We use .pt which loads the standard architecture automatically
model = YOLO("yolo11n.pt") 

# Train
results = model.train(
    data=f"{dataset_location}/data.yaml",
    epochs=100,             # As requested
    imgsz=640,
    batch=16,              # Standard batch size
    name="baseline_yolo11n_bccd",
    project="hemo_tests",
    plots=True             # Automatically generates comparison graphs
)

print("✅ Baseline Training Complete.")
print(f"📊 Results saved to: hemo_tests/baseline_yolo11n_bccd")
