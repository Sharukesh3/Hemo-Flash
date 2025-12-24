import os
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

# ==========================================
# 1. DOWNLOAD DATASET (BCCD)
# ==========================================
print("⬇️  Downloading BCCD Dataset...")
# Use the public BCCD project
rf = Roboflow(api_key="PjUym3maiIS3wU8Ce7zp") # <--- PASTE YOUR KEY HERE
project = rf.workspace("roboflow-100").project("bccd-ouzjz")
dataset = project.version(2).download("yolov11") # Downloads formatted for YOLO

# ==========================================
# 2. UPDATE CONFIG FOR TEST (nc: 3)
# ==========================================
# We need to temporarily change your 20-class model to 3 classes
# to match the BCCD dataset (RBC, WBC, Platelets)

hemo_yaml_path = "ultralytics/cfg/models/11/hemo-flash.yaml" # Path to your custom architecture
test_yaml_path = "hemo-flash-test.yaml"

with open(hemo_yaml_path, 'r') as f:
    config = yaml.safe_load(f)

# Modify for BCCD
config['nc'] = 3 
config['names'] = ['Platelets', 'RBC', 'WBC']

# Save temporary test config
with open(test_yaml_path, 'w') as f:
    yaml.dump(config, f)
print(f"✅ Created temporary config '{test_yaml_path}' with nc=3")

# ==========================================
# 3. TRAIN (SMOKE TEST)
# ==========================================
print("🚀 Starting Hemo-Flash Training Test...")

# Load the model using the modified architecture
model = YOLO(test_yaml_path)

# Train on the downloaded data
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=10,            # Short run just to test convergence
    imgsz=640,
    batch=8,
    name="hemo_flash_bccd_test",
    project="hemo_tests"
)

print("🎉 Test Complete! Check 'hemo_tests/hemo_flash_bccd_test' for results.")
