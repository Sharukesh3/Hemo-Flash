import os
import shutil
from roboflow import Roboflow

# ==========================================
# 1. SETUP
# ==========================================
# Your provided API Key
rf = Roboflow(api_key="PjUym3maiIS3wU8Ce7zp")

# Create a clean workspace folder
workspace_dir = "Blood_Roboflow_Workspace_Remastered"
os.makedirs(workspace_dir, exist_ok=True)
os.chdir(workspace_dir)

print(f"🚀 Starting download in: {os.getcwd()}")

# ==========================================
# 2. DOWNLOAD RAABIN (Version 4)
# ==========================================
try:
    print("\n⬇️  Downloading Raabin (Memoria)...")
    project = rf.workspace("memoria").project("raabin")
    version = project.version(4)
    dataset = version.download("yolov11")
    
    # Rename folder to be consistent
    if os.path.exists(dataset.location):
        # Remove old folder if exists to prevent conflicts
        if os.path.exists("Raabin_WBC"): shutil.rmtree("Raabin_WBC")
        os.rename(dataset.location, "Raabin_WBC")
        print("✅ Raabin secured in folder: Raabin_WBC")

except Exception as e:
    print(f"❌ Raabin Download Failed: {e}")

# ==========================================
# 3. DOWNLOAD BCCD (Platelets/RBC)
# ==========================================
try:
    print("\n⬇️  Downloading BCCD...")
    # Standard BCCD from Roboflow Universe
    project = rf.workspace("joseph-nelson").project("bccd")
    version = project.version(4) # v4 is usually stable
    dataset = version.download("yolov11")
    
    if os.path.exists(dataset.location):
        if os.path.exists("BCCD"): shutil.rmtree("BCCD")
        os.rename(dataset.location, "BCCD")
        print("✅ BCCD secured in folder: BCCD")

except Exception as e:
    print(f"❌ BCCD Download Failed: {e}")

# ==========================================
# 4. DOWNLOAD SICKLE CELL (Anemia) - UPDATED
# ==========================================
try:
    print("\n⬇️  Downloading Sickle Cell (researchmethodology-bwfx1)...")
    
    # --- UPDATED SOURCE ---
    project = rf.workspace("researchmethodology-bwfx1").project("sickle-cell-detector")
    version = project.version(2)
    dataset = version.download("yolov11")
    
    if os.path.exists(dataset.location):
        if os.path.exists("Sickle_Cell"): shutil.rmtree("Sickle_Cell")
        os.rename(dataset.location, "Sickle_Cell")
        print("✅ Sickle Cell secured in folder: Sickle_Cell")

except Exception as e:
    print(f"❌ Sickle Cell Download Failed: {e}")

print("\n-------------------------------------------------------------")
print("🎉 ALL DOWNLOADS FINISHED.")
print("   Location: Blood_Roboflow_Workspace/")
print("   Next Step: You MUST run the 'Class Remapper' script.")
print("   (Because '0' means different things in each folder!)")
print("-------------------------------------------------------------")