#!/bin/bash
# Extract SkyScenes dataset archives
# Handles both:
#   - .tar.gz files (from HuggingFace download)
#   - .png files that are actually tar archives (legacy format)
#
# Usage: bash scripts/extract_skyscenes.sh /path/to/SkyScenes

DATASET_ROOT="${1:-./SkyScenes}"
TEMP_DIR="/tmp/skyscenes_extract_$$"

echo "=============================================="
echo "SkyScenes Extractor"
echo "=============================================="
echo "Dataset path: $DATASET_ROOT"
echo "=============================================="

# Cleanup temp on exit
trap "rm -rf $TEMP_DIR" EXIT

is_tar() {
    file "$1" 2>/dev/null | grep -q "tar archive"
}

# Extract .tar.gz archive
extract_targz() {
    local archive="$1"
    local target_dir="$2"

    echo "[EXTRACT] $archive"
    mkdir -p "$TEMP_DIR"

    # Try extraction with different strip levels (archives have nested paths)
    tar -xzf "$archive" -C "$TEMP_DIR" 2>/dev/null || tar -xf "$archive" -C "$TEMP_DIR" 2>/dev/null

    # Move all PNG files to target, flattening structure
    find "$TEMP_DIR" -name "*.png" -type f -exec mv {} "$target_dir/" \; 2>/dev/null

    rm -rf "$TEMP_DIR"/*
    rm -f "$archive"
}

# Extract .png file that is actually a tar archive
extract_fake_png() {
    local tar_file="$1"
    local target_dir="$2"

    mkdir -p "$TEMP_DIR"
    tar -xf "$tar_file" -C "$TEMP_DIR" 2>/dev/null

    find "$TEMP_DIR" -name "*.png" -type f -exec mv {} "$target_dir/" \; 2>/dev/null

    rm -rf "$TEMP_DIR"/*
    rm -f "$tar_file"
}

# Process each data type
for data_type in Images Depth; do
    echo ""
    echo "Processing $data_type..."
    echo "----------------------------"

    data_dir="$DATASET_ROOT/$data_type"
    [ ! -d "$data_dir" ] && echo "  Not found: $data_dir" && continue

    # 1. Extract .tar.gz files
    while IFS= read -r archive; do
        [ -z "$archive" ] && continue
        target_dir=$(dirname "$archive")
        extract_targz "$archive" "$target_dir"
    done < <(find "$data_dir" -name "*.tar.gz" -type f 2>/dev/null)

    # 2. Extract .png files that are actually tar archives
    while IFS= read -r town_dir; do
        [ -z "$town_dir" ] && continue
        first_png=$(find "$town_dir" -maxdepth 1 -name "*.png" -type f 2>/dev/null | head -1)
        if [ -n "$first_png" ] && is_tar "$first_png"; then
            echo "[EXTRACT] tar-as-png in: $town_dir"
            for tar_file in "$town_dir"/*.png; do
                [ -f "$tar_file" ] && is_tar "$tar_file" && extract_fake_png "$tar_file" "$town_dir"
            done
        fi
    done < <(find "$data_dir" -type d -name "Town*" 2>/dev/null)
done

echo ""
echo "=============================================="
echo "Extraction complete!"
echo "=============================================="

# Verify
echo ""
echo "Verification:"
for data_type in Images Depth; do
    data_dir="$DATASET_ROOT/$data_type"
    [ ! -d "$data_dir" ] && continue
    count=$(find "$data_dir" -name "*.png" -type f 2>/dev/null | wc -l)
    sample=$(find "$data_dir" -name "*.png" -type f 2>/dev/null | head -1)
    if [ -n "$sample" ]; then
        ftype=$(file "$sample" | grep -o "PNG image data" || echo "NOT PNG")
        echo "  $data_type: $count files ($ftype)"
    else
        echo "  $data_type: no PNG files found"
    fi
done
