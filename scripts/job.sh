#!/bin/bash

# Run the job command passed as an argument when submitting the job ('python main.py' for example)
echo "Running command: $@"
srun uv run "$@"
