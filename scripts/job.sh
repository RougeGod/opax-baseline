#!/bin/bash

# Compute nodes have no internet access, so uv can only install from its cache.
# Keep it in $SCRATCH (not $HOME, which has a much smaller quota on most clusters).
export UV_CACHE_DIR="${SCRATCH:-$HOME}/uv-cache"
mkdir -p "$UV_CACHE_DIR"

# Run the job command passed as an argument when submitting the job ('python main.py' for example)
echo "Running command: $@"
srun uv run "$@" --seed "${SLURM_ARRAY_TASK_ID:-834}" #if not array, default to 834
