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

extracted=0
failed=0

# Find all .png files and check if they're tar archives
echo ""
echo "Scanning for tar archives..."

while IFS= read -r -d '' pngfile; do
    # Check if file is a tar archive
    filetype=$(file -b "$pngfile" 2>/dev/null)

    if echo "$filetype" | grep -qi "tar archive\|POSIX tar"; then
        target_dir=$(dirname "$pngfile")
        basename_file=$(basename "$pngfile")

        echo "Extracting: $pngfile"

        # Clear temp
        rm -rf "$TEMP_DIR"/*

        # Extract tar
        if tar -xf "$pngfile" -C "$TEMP_DIR" 2>/dev/null; then
            # Move all extracted PNGs to target dir
            find "$TEMP_DIR" -name "*.png" -type f | while read src; do
                mv "$src" "$target_dir/" 2>/dev/null
            done

            # Remove the tar file
            rm -f "$pngfile"
            extracted=$((extracted + 1))
        else
            echo "  FAILED to extract: $pngfile"
            failed=$((failed + 1))
        fi
    fi
done < <(find "$DATASET_ROOT" -name "*.png" -type f -print0 2>/dev/null)

# Also handle .tar.gz files
echo ""
echo "Scanning for .tar.gz files..."

while IFS= read -r -d '' archive; do
    target_dir=$(dirname "$archive")
    echo "Extracting: $archive"

    rm -rf "$TEMP_DIR"/*

    if tar -xzf "$archive" -C "$TEMP_DIR" 2>/dev/null; then
        find "$TEMP_DIR" -name "*.png" -type f | while read src; do
            mv "$src" "$target_dir/" 2>/dev/null
        done
        rm -f "$archive"
        extracted=$((extracted + 1))
    else
        echo "  FAILED: $archive"
        failed=$((failed + 1))
    fi
done < <(find "$DATASET_ROOT" -name "*.tar.gz" -type f -print0 2>/dev/null)

echo ""
echo "=============================================="
echo "Extraction Summary"
echo "=============================================="
echo "Extracted: $extracted archives"
echo "Failed: $failed archives"

# Verify
echo ""
echo "Verification:"
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
if [ "$failed" -eq 0 ]; then
    echo "SUCCESS!"
else
    echo "Completed with $failed errors"
fi
