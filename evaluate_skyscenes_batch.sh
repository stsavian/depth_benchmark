#!/bin/bash
#PBS -m bea
#PBS -M stefano.savian.ext@leonardocompany.com
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -j oe
#PBS -r n
#PBS -q gpu

#=============================================================================
# CONFIGURATION - Modify these variables to filter the dataset
#=============================================================================

# Model to evaluate (options: moge, moge-vitl, moge-2-vitl, moge-2-vitl-normal)
MODEL="moge-2-vitl"

# Altitudes in meters (space-separated, or empty for all)
# Available: 15 35 60
ALTITUDES="60"

# Pitch angles in degrees (space-separated, or empty for all)
# Available: 0 45 60 90
PITCHES="45"

# Weather/time of day conditions (space-separated, or empty for all)
# Available: ClearNoon, ClearSunset, CloudyNoon, CloudySunset, WetNoon, WetSunset, etc.
WEATHERS="ClearNoon"

# Towns (space-separated, or empty for all)
# Available: Town01, Town02, Town03, etc.
TOWNS="Town01"

# Maximum number of samples (empty for all)
MAX_SAMPLES=""

#=============================================================================
# END CONFIGURATION
#=============================================================================

echo "pbs dir"
echo  $PBS_O_WORKDIR
echo "bash dir"
echo $PWD
echo "cd to pbs dir"
cd $PBS_O_WORKDIR
echo $PWD

module load proxy/proxy_20
PROXY_ENVS="http_proxy=http://10.17.20.110:8080/"
PROXY_ENVS="${PROXY_ENVS},https_proxy=http://10.17.20.110:8080/"
PROXY_ENVS="${PROXY_ENVS},ftp_proxy=http://10.17.20.110:8080/"
PROXY_ENVS="${PROXY_ENVS},HTTP_PROXY=http://10.17.20.110:8080/"
PROXY_ENVS="${PROXY_ENVS},HTTPS_PROXY=http://10.17.20.110:8080/"
PROXY_ENVS="${PROXY_ENVS},FTP_PROXY=http://10.17.20.110:8080/"
BINDS="/archive,/cm,/davinci-1,/etc/resolv.conf,/run"

singularity_sif_path=/davinci-1/home/ssavian/container5
singularity_python=/opt/conda/envs/depth_benchmark/bin/python

cd /davinci-1/home/ssavian/CODE/depth_benchmark

# Build optional arguments
EVAL_ARGS=""
if [ -n "$ALTITUDES" ]; then
    EVAL_ARGS="$EVAL_ARGS --altitudes $ALTITUDES"
fi
if [ -n "$PITCHES" ]; then
    EVAL_ARGS="$EVAL_ARGS --pitches $PITCHES"
fi
if [ -n "$WEATHERS" ]; then
    EVAL_ARGS="$EVAL_ARGS --weathers $WEATHERS"
fi
if [ -n "$TOWNS" ]; then
    EVAL_ARGS="$EVAL_ARGS --towns $TOWNS"
fi
if [ -n "$MAX_SAMPLES" ]; then
    EVAL_ARGS="$EVAL_ARGS --max-samples $MAX_SAMPLES"
fi

OUTPUT_DIR="results/skyscenes_${MODEL}"

echo "=============================================="
echo "SkyScenes Evaluation"
echo "=============================================="
echo "Model: $MODEL"
echo "Altitudes: ${ALTITUDES:-all}"
echo "Pitches: ${PITCHES:-all}"
echo "Weathers: ${WEATHERS:-all}"
echo "Towns: ${TOWNS:-all}"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Run evaluation
singularity exec --nv -B $BINDS --env PATH="\$PATH:$PATH",${PROXY_ENVS} $singularity_sif_path \
    $singularity_python scripts/evaluate_skyscenes.py \
    --dataset /davinci-1/home/ssavian/DATASETS/SkyScenes \
    --model $MODEL \
    --output $OUTPUT_DIR \
    --device cuda \
    $EVAL_ARGS

# Plot error vs depth
singularity exec --nv -B $BINDS --env PATH="\$PATH:$PATH",${PROXY_ENVS} $singularity_sif_path \
    $singularity_python scripts/plot_error_vs_depth.py \
    --csv ${OUTPUT_DIR}/results_per_sample.csv \
    --output ${OUTPUT_DIR}/plots
