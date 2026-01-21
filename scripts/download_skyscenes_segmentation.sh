#!/bin/bash
# Download SkyScenes segmentation masks from Hugging Face
# Note: Segmentation only available for ClearNoon weather
#
# Usage: bash scripts/download_skyscenes_segmentation.sh [options]
#
# Options:
#   --path PATH      Download/extract path (default: ./SkyScenes)
#   --no-extract     Skip extraction after download

set -e

# Default settings
DOWNLOAD_FOLDER="/davinci-1/home/ssavian/DATASETS/SkyScenes"
EXTRACT=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --path) DOWNLOAD_FOLDER="$2"; shift 2 ;;
        --no-extract) EXTRACT=false; shift ;;
        *) shift ;;
    esac
done

# Height and Pitch variations (12 total)
HP=('H_15_P_0' 'H_15_P_45' 'H_15_P_60' 'H_15_P_90' 'H_35_P_0' 'H_35_P_45' 'H_35_P_60' 'H_35_P_90' 'H_60_P_0' 'H_60_P_45' 'H_60_P_60' 'H_60_P_90')

# Segmentation only available for ClearNoon
WEATHER_SEGMENT=('ClearNoon')

# Town layouts
TOWNS=('Town01' 'Town02' 'Town03' 'Town04' 'Town05' 'Town06' 'Town07' 'Town10HD')

# Base URL
BASE_URL="https://huggingface.co/datasets/hoffman-lab/SkyScenes/resolve/main"

echo "=============================================="
echo "SkyScenes Segmentation Data Downloader"
echo "=============================================="
echo "Download folder: $DOWNLOAD_FOLDER"
echo "Note: Segmentation only available for ClearNoon weather"
echo "=============================================="

download_segmentation() {
    for hp in "${HP[@]}"; do
        for weather in "${WEATHER_SEGMENT[@]}"; do
            for town in "${TOWNS[@]}"; do
                download_url="${BASE_URL}/Segment/${hp}/${weather}/${town}/${town}.tar.gz"
                download_folder="${DOWNLOAD_FOLDER}/Segment/${hp}/${weather}/${town}"
                output_file="${download_folder}/${town}.tar.gz"

                # Skip if already downloaded
                if [ -f "$output_file" ]; then
                    echo "[SKIP] Already exists: $output_file"
                    continue
                fi

                mkdir -p "$download_folder"
                echo "[DOWNLOAD] Segment / $hp / $weather / $town"
                wget -q --show-progress -O "$output_file" "$download_url" || {
                    echo "[ERROR] Failed to download: $download_url"
                    rm -f "$output_file"
                }
            done
        done
    done
}

# Download segmentation data
echo "Downloading segmentation masks..."
download_segmentation

echo ""
echo "=============================================="
echo "Download complete!"
echo "=============================================="

# Extract archives if requested
if [ "$EXTRACT" = true ]; then
    echo ""
    echo "Extracting tar.gz archives..."
    echo "=============================================="

    for hp in "${HP[@]}"; do
        for weather in "${WEATHER_SEGMENT[@]}"; do
            for town in "${TOWNS[@]}"; do
                archive="${DOWNLOAD_FOLDER}/Segment/${hp}/${weather}/${town}/${town}.tar.gz"
                target_dir="${DOWNLOAD_FOLDER}/Segment/${hp}/${weather}/${town}"

                if [ -f "$archive" ]; then
                    echo "[EXTRACT] Segment / $hp / $weather / $town"
                    tar -xzf "$archive" -C "$target_dir" --strip-components=8 2>/dev/null || \
                    tar -xzf "$archive" -C "$target_dir" --strip-components=7 2>/dev/null || \
                    tar -xzf "$archive" -C "$target_dir" 2>/dev/null
                    rm "$archive"
                fi
            done
        done
    done

    echo ""
    echo "Extraction complete!"
fi

echo ""
echo "Segmentation data ready at: $DOWNLOAD_FOLDER/Segment/"
