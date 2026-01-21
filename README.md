# Depth Benchmark

Benchmark monocular depth estimation and segmentation models on aerial imagery (SkyScenes dataset).

## Installation

```bash
conda env create -f environment.yml
conda activate depth_benchmark
pip install -e .

# For segmentation (optional)
pip install transformers
```

## Download SkyScenes

```bash
# Download images + depth
bash scripts/download_skyscenes.sh --path /path/to/SkyScenes

# Download segmentation masks (ClearNoon only)
bash scripts/download_skyscenes_segmentation.sh --path /path/to/SkyScenes

# Extract all archives (images, depth, segmentation)
bash scripts/extract_skyscenes.sh /path/to/SkyScenes
```

## Depth Benchmarking

```bash
python scripts/evaluate_skyscenes.py \
    --dataset /path/to/SkyScenes \
    --model moge \
    --output results/

# Filter by altitude/pitch/weather/town
python scripts/evaluate_skyscenes.py \
    --dataset /path/to/SkyScenes \
    --model moge \
    --altitudes 15 30 \
    --pitches 0 -45 \
    --weathers ClearNoon \
    --towns Town01 Town02
```

## Segmentation Benchmarking

```bash
python scripts/evaluate_skyscenes_segmentation.py \
    --dataset /path/to/SkyScenes \
    --model segformer-b5-cityscapes \
    --output results/

# Filter by altitude/pitch/town
python scripts/evaluate_skyscenes_segmentation.py \
    --dataset /path/to/SkyScenes \
    --model segformer-b5-cityscapes \
    --altitudes 15 \
    --pitches 0 \
    --towns Town01
```

## Available Models

**Depth**: `moge`, `moge-2-vitl`

**Segmentation**: `segformer-b0-cityscapes`, `segformer-b5-cityscapes`
