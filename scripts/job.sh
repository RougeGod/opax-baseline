#!/bin/bash
#SBATCH --output=logs/%j/slurm-%j.out

# Run the job command passed as an argument when submitting the job ('python main.py' for example)
echo "Running command: $@"
srun uv run "$@"
