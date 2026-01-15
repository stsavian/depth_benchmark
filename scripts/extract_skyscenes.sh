#!/bin/bash
# Script to extract SkyScenes tar archives
# The dataset downloads files as .png but they are actually tar archives

DATASET_ROOT="${1:-/home/ssavian/DATASETS/SkyScenes}"
TEMP_DIR="/tmp/skyscenes_extract"

echo "Extracting SkyScenes dataset from: $DATASET_ROOT"
echo "================================================"

# Function to check if file is a tar archive
is_tar() {
    file "$1" | grep -q "tar archive"
}

# Function to extract tar file with correct path handling
extract_tar() {
    local tar_file="$1"
    local target_dir="$2"

    echo "Extracting: $tar_file"

    # Create temp extraction directory
    mkdir -p "$TEMP_DIR"

    # Extract to temp
    tar -xf "$tar_file" -C "$TEMP_DIR" 2>/dev/null

    # Find the extracted files and move them to the correct location
    # The tar contains full paths like: srv/hoffman-lab/flash9/.../filename.png
    # We need to strip the prefix and keep just the filename
    find "$TEMP_DIR" -name "*.png" -type f | while read src_file; do
        filename=$(basename "$src_file")
        mv "$src_file" "$target_dir/$filename" 2>/dev/null
    done

    # Remove the tar file (which is named .png)
    rm "$tar_file"

    # Cleanup temp
    rm -rf "$TEMP_DIR"
}

# Process Images folder
echo ""
echo "Processing Images folder..."
echo "----------------------------"
for hp_dir in "$DATASET_ROOT"/Images/H_*_P_*/; do
    for weather_dir in "$hp_dir"*/; do
        for town_dir in "$weather_dir"*/; do
            # Check first file in directory
            first_file=$(ls "$town_dir"*.png 2>/dev/null | head -1)
            if [ -n "$first_file" ] && is_tar "$first_file"; then
                echo "Found tar archives in: $town_dir"
                for tar_file in "$town_dir"*.png; do
                    if is_tar "$tar_file"; then
                        extract_tar "$tar_file" "$town_dir"
                    fi
                done
            fi
        done
    done
done

# Process Depth folder
echo ""
echo "Processing Depth folder..."
echo "----------------------------"
for hp_dir in "$DATASET_ROOT"/Depth/H_*_P_*/; do
    for weather_dir in "$hp_dir"*/; do
        for town_dir in "$weather_dir"*/; do
            # Check first file in directory
            first_file=$(ls "$town_dir"*.png 2>/dev/null | head -1)
            if [ -n "$first_file" ] && is_tar "$first_file"; then
                echo "Found tar archives in: $town_dir"
                for tar_file in "$town_dir"*.png; do
                    if is_tar "$tar_file"; then
                        extract_tar "$tar_file" "$town_dir"
                    fi
                done
            fi
        done
    done
done

echo ""
echo "Extraction complete!"
echo "===================="

# Verify extraction
echo ""
echo "Verifying extraction..."
for hp_dir in "$DATASET_ROOT"/Images/H_*_P_*/; do
    hp_name=$(basename "$hp_dir")
    sample=$(find "$hp_dir" -name "*.png" -type f 2>/dev/null | head -1)
    if [ -n "$sample" ]; then
        ftype=$(file "$sample" | grep -o "PNG image data" || echo "NOT PNG")
        echo "$hp_name Images: $ftype"
    fi
done
