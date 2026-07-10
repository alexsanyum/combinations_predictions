#!/bin/bash
#SBATCH --job-name=concat_embs
#SBATCH --nodes=1
#SBATCH --partition=sequana_cpu_dev       
#SBATCH --ntasks-per-node=16                  
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --chdir=/scratch/pcmrnbio2/alex.yumbo/combinations_predictions
#SBATCH --output=/scratch/pcmrnbio2/alex.yumbo/logs/job_%j.out
#SBATCH --error=/scratch/pcmrnbio2/alex.yumbo/logs/job_%j.err

REPO_DIR="/scratch/pcmrnbio2/alex.yumbo/combinations_predictions"
CONTAINER_IMG="/scratch/pcmrnbio2/alex.yumbo/containers/py_data.sif"

echo "=== CONTAINER SANITY CHECK ==="

# 1. Verify python path and version inside the container
echo "Checking Python path inside container:"
singularity exec $CONTAINER_IMG which python

echo "Checking Python version inside container:"
singularity exec $CONTAINER_IMG python --version

# 2. Test running code and checking directory bindings
echo "Testing repository access and Python execution:"
singularity exec \
    -B $REPO_DIR:/app \
    $CONTAINER_IMG \
    python -c "import os; print('Hello from the container!'); print('Available files in /app:', os.listdir('/app'))"

echo "=== CHECK COMPLETE ==="