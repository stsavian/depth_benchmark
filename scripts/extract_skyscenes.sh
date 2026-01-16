#!/bin/bash
# Extract SkyScenes dataset
# Handles .png files that are actually tar archives (HuggingFace download format)
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

# Find all .png files that are tar archives
echo ""
echo "Scanning for tar archives disguised as .png..."

for pngfile in $(find "$DATASET_ROOT" -name "*.png" -type f 2>/dev/null); do
    filetype=$(file -b "$pngfile" 2>/dev/null)

    if echo "$filetype" | grep -qi "tar archive\|POSIX tar"; then
        target_dir=$(dirname "$pngfile")
        echo "Extracting: $pngfile"

        # Clear temp
        rm -rf "$TEMP_DIR"/*

        # Extract tar (show errors)
        if tar -xf "$pngfile" -C "$TEMP_DIR"; then
            # Move all extracted PNGs to target dir
            for src in $(find "$TEMP_DIR" -name "*.png" -type f); do
                mv "$src" "$target_dir/"
            done
            # Remove the tar file
            rm -f "$pngfile"
            echo "  OK"
        else
            echo "  FAILED"
        fi
    fi
done

# Handle .tar.gz files (try both gzip and plain tar)
echo ""
echo "Scanning for .tar.gz files..."

for archive in $(find "$DATASET_ROOT" -name "*.tar.gz" -type f 2>/dev/null); do
    target_dir=$(dirname "$archive")
    echo "Extracting: $archive"

    rm -rf "$TEMP_DIR"/*

    # Try gzip first, then plain tar
    if tar -xzf "$archive" -C "$TEMP_DIR" 2>/dev/null || tar -xf "$archive" -C "$TEMP_DIR"; then
        for src in $(find "$TEMP_DIR" -name "*.png" -type f); do
            mv "$src" "$target_dir/"
        done
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
for dtype in Images Depth; do
    dpath="$DATASET_ROOT/$dtype"
    [ ! -d "$dpath" ] && continue

    count=$(find "$dpath" -name "*.png" -type f 2>/dev/null | wc -l)
    sample=$(find "$dpath" -name "*.png" -type f 2>/dev/null | head -1)

    if [ -n "$sample" ]; then
        ftype=$(file -b "$sample" | head -c 30)
        echo "$dtype: $count files ($ftype)"
    else
        echo "$dtype: 0 files"
    fi
done

echo ""
echo "Done!"
