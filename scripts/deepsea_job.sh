#!/bin/bash

# Compute nodes have no internet access, so uv can only install from its cache.
# Keep it in $SCRATCH (not $HOME, which has a much smaller quota on most clusters).
export UV_CACHE_DIR="${SCRATCH:-$HOME}/uv-cache"
mkdir -p "$UV_CACHE_DIR"

# Run the job command passed as an argument when submitting the job ('python main.py' for example)
echo "Running command: $@"
parallel uv run "$@" --seed {3} --size {1} --horizon {1} --total_train_steps {2} ::: $(seq 25 5 50) :::+ $(seq 250000 50000 500000) ::: $(seq 50 79)
