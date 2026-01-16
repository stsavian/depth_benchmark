# Depth Benchmark

Benchmark monocular depth estimation models on aerial imagery (SkyScenes dataset).

## Setup

```bash
conda env create -f environment.yml
conda activate depth_benchmark
pip install -e .
```

## Download SkyScenes

```bash
# Download dataset from HuggingFace (images + depth)
bash scripts/download_skyscenes.sh --path /path/to/SkyScenes

# Or download only depth maps (smaller)
bash scripts/download_skyscenes.sh --path /path/to/SkyScenes --depth-only

# Extract existing archives (if downloaded separately)
bash scripts/extract_skyscenes.sh /path/to/SkyScenes
```

## Quick Start

```bash
python scripts/evaluate_skyscenes.py --dataset /path/to/SkyScenes --model moge
```

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/evaluate_skyscenes.py` | Evaluate depth model on SkyScenes, grouped by altitude/pitch |
| `scripts/extract_skyscenes.sh` | Extract SkyScenes tar archives to PNG files |

## Key Options

```bash
python scripts/evaluate_skyscenes.py \
    --dataset /path/to/SkyScenes \
    --model moge \
    --output results/ \
    --max-samples 50  # Quick test with fewer samples
```

## Dataset Note

SkyScenes includes slight jitter in height values (Δh ~ N(1, 2.5m)) to simulate realistic UAV actuation imperfections.
