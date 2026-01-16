#!/bin/bash
# Extract SkyScenes dataset
# Handles .png files that are actually tar archives (HuggingFace download format)
# Each .png tar contains ALL real PNGs for that folder with absolute paths
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

# Track processed directories to avoid re-extracting identical tars
declare -A processed_dirs

# Find all .png files that are tar archives
echo ""
echo "Scanning for tar archives disguised as .png..."

# Use while read to handle filenames with spaces
find "$DATASET_ROOT" -name "*.png" -type f 2>/dev/null | while read -r pngfile; do
    filetype=$(file -b "$pngfile" 2>/dev/null)

    if echo "$filetype" | grep -qi "tar archive\|POSIX tar"; then
        target_dir=$(dirname "$pngfile")

        # Check if we already processed this directory
        if [ -f "$target_dir/.extracted" ]; then
            # Just remove this duplicate tar
            rm -f "$pngfile"
            continue
        fi

        echo "Extracting: $pngfile"
        echo "  Target: $target_dir"

        # Clear temp
        rm -rf "$TEMP_DIR"/*

        # Extract tar with --strip-components to handle absolute paths
        # The tar contains paths like: srv/hoffman-lab/.../SkyScenes/Images/.../file.png
        # We need to extract just the .png files to the target directory
        if tar -xf "$pngfile" -C "$TEMP_DIR" 2>/dev/null; then
            # Find and move all extracted PNGs (handles nested absolute paths)
            extracted_count=0
            while IFS= read -r src; do
                basename_file=$(basename "$src")
                mv "$src" "$target_dir/$basename_file"
                ((extracted_count++))
            done < <(find "$TEMP_DIR" -name "*.png" -type f 2>/dev/null)

            echo "  Extracted $extracted_count PNG files"

            # Mark directory as processed
            touch "$target_dir/.extracted"

            # Remove the tar file
            rm -f "$pngfile"
            echo "  OK"
        else
            echo "  FAILED to extract"
        fi
    fi
done

# Now remove all remaining tar-disguised-as-png files (duplicates)
echo ""
echo "Cleaning up duplicate tar files..."
find "$DATASET_ROOT" -name "*.png" -type f 2>/dev/null | while read -r pngfile; do
    filetype=$(file -b "$pngfile" 2>/dev/null)
    if echo "$filetype" | grep -qi "tar archive\|POSIX tar"; then
        echo "  Removing duplicate: $(basename "$pngfile")"
        rm -f "$pngfile"
    fi
done

# Clean up .extracted markers
find "$DATASET_ROOT" -name ".extracted" -type f -delete 2>/dev/null

# Handle .tar.gz files (try both gzip and plain tar)
echo ""
echo "Scanning for .tar.gz files..."

find "$DATASET_ROOT" -name "*.tar.gz" -type f 2>/dev/null | while read -r archive; do
    target_dir=$(dirname "$archive")
    echo "Extracting: $archive"

    rm -rf "$TEMP_DIR"/*

    # Try gzip first, then plain tar
    if tar -xzf "$archive" -C "$TEMP_DIR" 2>/dev/null || tar -xf "$archive" -C "$TEMP_DIR" 2>/dev/null; then
        extracted_count=0
        while IFS= read -r src; do
            basename_file=$(basename "$src")
            mv "$src" "$target_dir/$basename_file"
            ((extracted_count++))
        done < <(find "$TEMP_DIR" -name "*.png" -type f 2>/dev/null)

        echo "  Extracted $extracted_count PNG files"
        rm -f "$archive"
        echo "  OK"
    else
        echo "  FAILED"
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
