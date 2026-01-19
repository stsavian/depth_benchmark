#!/bin/bash
# Download SkyScenes dataset from Hugging Face
# Usage: bash scripts/download_skyscenes.sh [options]
#
# Options:
#   --path PATH      Download/extract path (default: ./SkyScenes)
#   --images-only    Download only RGB images
#   --depth-only     Download only depth maps
#   --all            Download images + depth (default)
#   --no-extract     Skip extraction after download
#   --extract-only   Only extract existing archives (no download)

set -e

# Default settings
DOWNLOAD_FOLDER="/davinci-1/home/ssavian/DATASETS/SkyScenes"
MODE="all"
EXTRACT=true
EXTRACT_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --path) DOWNLOAD_FOLDER="$2"; shift 2 ;;
        --images-only) MODE="images"; shift ;;
        --depth-only) MODE="depth"; shift ;;
        --all) MODE="all"; shift ;;
        --no-extract) EXTRACT=false; shift ;;
        --extract-only) EXTRACT_ONLY=true; shift ;;
        *) shift ;;
    esac
done

# Height and Pitch variations (12 total)
HP=('H_15_P_0' 'H_15_P_45' 'H_15_P_60' 'H_15_P_90' 'H_35_P_0' 'H_35_P_45' 'H_35_P_60' 'H_35_P_90' 'H_60_P_0' 'H_60_P_45' 'H_60_P_60' 'H_60_P_90')
#HP=('H_15_P_0' 'H_15_P_45' )
# Weather conditions (depth only uses ClearNoon)
WEATHER_IMAGES=('ClearNoon' 'ClearNight' 'ClearSunset' 'CloudyNoon' 'MidRainyNoon')
#WEATHER_IMAGES=('ClearNoon' )
WEATHER_DEPTH=('ClearNoon')

# Town layouts
TOWNS=('Town01' 'Town02' 'Town03' 'Town04' 'Town05' 'Town06' 'Town07' 'Town10HD')
#TOWNS=('Town01')
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

# Extract archives if requested
if [ "$EXTRACT" = true ]; then
    echo ""
    echo "Extracting tar.gz archives..."
    echo "=============================================="

    extract_archives() {
        local data_type=$1
        for hp in "${HP[@]}"; do
            for town in "${TOWNS[@]}"; do
                # For depth, only ClearNoon exists
                if [ "$data_type" = "Depth" ]; then
                    weather_list=("ClearNoon")
                else
                    weather_list=("${WEATHER_IMAGES[@]}")
                fi

                for weather in "${weather_list[@]}"; do
                    archive="${DOWNLOAD_FOLDER}/${data_type}/${hp}/${weather}/${town}/${town}.tar.gz"
                    target_dir="${DOWNLOAD_FOLDER}/${data_type}/${hp}/${weather}/${town}"

                    if [ -f "$archive" ]; then
                        echo "[EXTRACT] $data_type / $hp / $weather / $town"
                        tar -xzf "$archive" -C "$target_dir" --strip-components=8 2>/dev/null || \
                        tar -xzf "$archive" -C "$target_dir" --strip-components=7 2>/dev/null || \
                        tar -xzf "$archive" -C "$target_dir" 2>/dev/null
                        rm "$archive"
                    fi
                done
            done
        done
    }

    case "$MODE" in
        --images-only) extract_archives "Images" ;;
        --depth-only) extract_archives "Depth" ;;
        --all|*)
            extract_archives "Images"
            extract_archives "Depth"
            ;;
    esac

    echo ""
    echo "Extraction complete!"
fi

# echo ""
# echo "Run benchmark:"
# echo "  /opt/conda/envs/depth_benchmark/bin/python scripts/evaluate_skyscenes.py --dataset $DOWNLOAD_FOLDER"
