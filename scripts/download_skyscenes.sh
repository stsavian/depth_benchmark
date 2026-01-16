#!/bin/bash
# Download SkyScenes dataset from Hugging Face
# Usage: bash scripts/download_skyscenes.sh /path/to/download/folder
#
# Options:
#   --images-only    Download only RGB images
#   --depth-only     Download only depth maps
#   --all            Download images + depth (default)

set -e

# Default settings
DOWNLOAD_FOLDER="${1:-./SkyScenes}"
MODE="${2:---all}"

# Height and Pitch variations (12 total)
HP=('H_15_P_0' 'H_15_P_45' 'H_15_P_60' 'H_15_P_90' 'H_35_P_0' 'H_35_P_45' 'H_35_P_60' 'H_35_P_90' 'H_60_P_0' 'H_60_P_45' 'H_60_P_60' 'H_60_P_90')

# Weather conditions (depth only uses ClearNoon)
WEATHER_IMAGES=('ClearNoon' 'ClearNight' 'ClearSunset' 'CloudyNoon' 'MidRainyNoon')
WEATHER_DEPTH=('ClearNoon')

# Town layouts
TOWNS=('Town01' 'Town02' 'Town03' 'Town04' 'Town05' 'Town06' 'Town07' 'Town10HD')

# Base URL
BASE_URL="https://huggingface.co/datasets/hoffman-lab/SkyScenes/resolve/main"

echo "=============================================="
echo "SkyScenes Dataset Downloader"
echo "=============================================="
echo "Download folder: $DOWNLOAD_FOLDER"
echo "Mode: $MODE"
echo "=============================================="

download_subset() {
    local data_type=$1  # Images or Depth
    local weather_list=("${!2}")

    for hp in "${HP[@]}"; do
        for weather in "${weather_list[@]}"; do
            for town in "${TOWNS[@]}"; do
                download_url="${BASE_URL}/${data_type}/${hp}/${weather}/${town}/${town}.tar.gz"
                download_folder="${DOWNLOAD_FOLDER}/${data_type}/${hp}/${weather}/${town}"
                output_file="${download_folder}/${town}.tar.gz"

                # Skip if already downloaded
                if [ -f "$output_file" ]; then
                    echo "[SKIP] Already exists: $output_file"
                    continue
                fi

                mkdir -p "$download_folder"
                echo "[DOWNLOAD] $data_type / $hp / $weather / $town"
                wget -q --show-progress -O "$output_file" "$download_url" || {
                    echo "[ERROR] Failed to download: $download_url"
                    rm -f "$output_file"
                }
            done
        done
    done
}

# Download based on mode
case "$MODE" in
    --images-only)
        echo "Downloading RGB images only..."
        download_subset "Images" WEATHER_IMAGES[@]
        ;;
    --depth-only)
        echo "Downloading depth maps only..."
        download_subset "Depth" WEATHER_DEPTH[@]
        ;;
    --all|*)
        echo "Downloading images and depth maps..."
        download_subset "Images" WEATHER_IMAGES[@]
        download_subset "Depth" WEATHER_DEPTH[@]
        ;;
esac

echo ""
echo "=============================================="
echo "Download complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Extract the archives:"
echo "     bash scripts/extract_skyscenes.sh $DOWNLOAD_FOLDER"
echo ""
echo "  2. Run benchmark:"
echo "     python scripts/evaluate_skyscenes.py --dataset $DOWNLOAD_FOLDER"
