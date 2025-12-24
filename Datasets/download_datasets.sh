#!/bin/bash
# Create Main Workspace
mkdir -p Blood_Workspace
cd Blood_Workspace

# =========================================================
# 1. RAABIN-WBC (Recursive Scraping)
# =========================================================
echo "🚀 Starting Raabin Bulk Download..."

# Function to handle recursive scraping and unzipping
download_raabin_folder() {
    local TARGET_URL=$1
    local SAVE_DIR=$2

    echo "   ⬇️  Scraping: $TARGET_URL"
    mkdir -p "$SAVE_DIR"
    cd "$SAVE_DIR"

    # Wget Magic:
    # -r: Recursive | -np: No Parent | -nd: No Directories (flatten)
    # -A zip: Only download zip files
    # -P .: Save to current directory
    wget -r -np -nd -A zip "$TARGET_URL"

    echo "   📦 Extracting $SAVE_DIR..."
    for zipfile in *.zip; do
        [ -e "$zipfile" ] || continue
        unzip -q -o "$zipfile"
        # Optional: rm "$zipfile" # Uncomment to save space
    done
    cd ../..
}

# --- Download First Microscope (Camera A) ---
download_raabin_folder "https://dl.raabindata.com/WBC/First_microscope/" "Raabin_First_Microscope"

# --- Download Second Microscope (Camera B) ---
download_raabin_folder "https://dl.raabindata.com/WBC/Second_microscope/" "Raabin_Second_Microscope"


# =========================================================
# 2. BCCD (Platelets + RBCs)
# =========================================================
echo "⬇️  Downloading BCCD (Standard)..."
mkdir -p BCCD
cd BCCD
git clone https://github.com/Shenggan/BCCD_Dataset.git .
rm -rf export
cd ..

# =========================================================
# 3. SICKLE CELL (Anemia)
# =========================================================
echo "⬇️  Downloading Sickle Cell (Kaggle)..."
mkdir -p Sickle_Cell
cd Sickle_Cell
# Ensure your kaggle.json is in ~/.kaggle/
kaggle datasets download -d florencetushabe/sickle-cell-disease-dataset
unzip -q sickle-cell-disease-dataset.zip
rm sickle-cell-disease-dataset.zip
cd ..

# =========================================================
# FINAL STATUS
# =========================================================
echo "✅ ALL DOWNLOADS COMPLETE."
echo "---------------------------------------------------"
echo "📂 Data Structure:"
echo "   /Blood_Workspace"
echo "    ├── Raabin_First_Microscope/  (Extracted Slides)"
echo "    ├── Raabin_Second_Microscope/ (Extracted Slides)"
echo "    ├── BCCD/                     (Images/Annotations)"
echo "    └── Sickle_Cell/              (Kaggle Data)"
echo "---------------------------------------------------"
