#!/bin/bash
#SBATCH --time=5:59:59
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=aip-amw8
#SBATCH --gpus=1


# Compute nodes have no internet access, so uv can only install from its cache.
# Keep it in $SCRATCH (not $HOME, which has a much smaller quota on most clusters).
export UV_CACHE_DIR="${SCRATCH:-$HOME}/uv-cache"
mkdir -p "$UV_CACHE_DIR"
export UV_OFFLINE=1
export WANDB_MODE=offline

# Run the job command passed as an argument when submitting the job ('python main.py' for example)
echo "Running command: $@"
parallel srun uv run experiments/pendulum_exp/active_exploration_exp_pendulum.py --exp_name PendVulc --use_wandb --rollout_steps 20 --total_train_steps 5000 --seed {} ::: $(seq 1000 1029)
