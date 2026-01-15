# Depth Benchmark

Benchmark monocular depth estimation models on aerial imagery (SkyScenes dataset).

## Setup

```bash
conda activate depth_benchmark
pip install -e .
```

## Quick Start

```bash
# Extract SkyScenes tar archives (run once)
bash scripts/extract_skyscenes.sh /path/to/SkyScenes

# Run benchmark
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
