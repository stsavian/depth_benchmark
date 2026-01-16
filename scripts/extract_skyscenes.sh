#!/bin/bash
# Extract SkyScenes dataset
# Handles nested archives: .tar.gz -> .png (tar) -> real PNGs
# Usage: bash scripts/extract_skyscenes.sh /path/to/SkyScenes

DATASET_ROOT="${1:-.}"

echo "=============================================="
echo "SkyScenes Extractor"
echo "Path: $DATASET_ROOT"
echo "=============================================="

if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: Directory not found: $DATASET_ROOT"
    exit 1
fi

# Create temp dir
TEMP_DIR=$(mktemp -d)
echo "Temp dir: $TEMP_DIR"

cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Function to extract tar-disguised-as-png files
extract_tar_pngs() {
    local found=0

    # Get list of directories containing tar-disguised-as-png files
    local tar_png_dirs
    tar_png_dirs=$(find "$DATASET_ROOT" -name "*.png" -type f -exec sh -c 'file -b "$1" | grep -qi "tar archive\|POSIX tar" && dirname "$1"' _ {} \; 2>/dev/null | sort -u)

    for target_dir in $tar_png_dirs; do
        [ -z "$target_dir" ] && continue

        # Find the first tar-png in this directory
        local first_tar=""
        for f in "$target_dir"/*.png; do
            [ -f "$f" ] || continue
            if file -b "$f" 2>/dev/null | grep -qi "tar archive\|POSIX tar"; then
                first_tar="$f"
                break
            fi
        done

        [ -z "$first_tar" ] && continue
        found=1

        echo "Extracting tar-png: $first_tar"

        # Clear temp
        rm -rf "$TEMP_DIR"/*

        # Extract tar
        if tar -xf "$first_tar" -C "$TEMP_DIR" 2>/dev/null; then
            # Find and move all extracted PNGs (handles nested absolute paths)
            local extracted_count=0
            while IFS= read -r src; do
                local basename_file
                basename_file=$(basename "$src")
                mv "$src" "$target_dir/$basename_file"
                ((extracted_count++))
            done < <(find "$TEMP_DIR" -name "*.png" -type f 2>/dev/null)

            echo "  Extracted $extracted_count files -> OK"
        else
            echo "  FAILED to extract"
        fi

        # Remove ALL tar-png files in this directory
        for f in "$target_dir"/*.png; do
            [ -f "$f" ] || continue
            if file -b "$f" 2>/dev/null | grep -qi "tar archive\|POSIX tar"; then
                rm -f "$f"
            fi
        done
    done

    return $found
}

# STEP 1: Handle .tar.gz files first
echo ""
echo "=== Step 1: Extracting .tar.gz files ==="

find "$DATASET_ROOT" -name "*.tar.gz" -type f 2>/dev/null | while read -r archive; do
    target_dir=$(dirname "$archive")
    echo "Extracting: $archive"

    rm -rf "$TEMP_DIR"/*

    # Try gzip first, then plain tar (some .tar.gz are actually just .tar)
    if tar -xzf "$archive" -C "$TEMP_DIR" 2>/dev/null || tar -xf "$archive" -C "$TEMP_DIR" 2>/dev/null; then
        extracted_count=0
        while IFS= read -r src; do
            basename_file=$(basename "$src")
            mv "$src" "$target_dir/$basename_file"
            ((extracted_count++))
        done < <(find "$TEMP_DIR" -name "*.png" -type f 2>/dev/null)

        echo "  Extracted $extracted_count files"
        rm -f "$archive"
        echo "  OK"
    else
        echo "  FAILED"
    fi
done

# STEP 2: Extract tar-disguised-as-png files (loop until none remain)
echo ""
echo "=== Step 2: Extracting tar-disguised-as-png files ==="

pass=1
while true; do
    echo "Pass $pass..."
    if ! extract_tar_pngs; then
        echo "No more tar-png files found."
        break
    fi
    ((pass++))
    # Safety limit
    if [ $pass -gt 10 ]; then
        echo "WARNING: Exceeded 10 passes, stopping."
        break
    fi
done

# Verify
echo ""
echo "=============================================="
echo "Verification"
echo "=============================================="
for dtype in Images Depth Segment Instance; do
    dpath="$DATASET_ROOT/$dtype"
    [ ! -d "$dpath" ] && continue

    # Count only real PNG files (not tars)
    real_count=0
    tar_count=0
    sample=""

    while IFS= read -r f; do
        ftype=$(file -b "$f" 2>/dev/null)
        if echo "$ftype" | grep -qi "PNG image"; then
            ((real_count++))
            [ -z "$sample" ] && sample="$f"
        elif echo "$ftype" | grep -qi "tar archive\|POSIX tar"; then
            ((tar_count++))
        fi
    done < <(find "$dpath" -name "*.png" -type f 2>/dev/null)

    if [ -n "$sample" ]; then
        ftype=$(file -b "$sample" | head -c 40)
        echo "$dtype: $real_count real PNGs ($ftype)"
    else
        echo "$dtype: $real_count real PNGs"
    fi

    if [ $tar_count -gt 0 ]; then
        echo "  WARNING: $tar_count .png files are still tar archives!"
    fi
done

echo ""
echo "Done!"
