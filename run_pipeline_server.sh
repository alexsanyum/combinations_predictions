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


echo "Test if is accessing to the repository..."

singularity exec \
    -B $REPO_DIR:/app \
    $CONTAINER_IMG \
    python "print('Hello from the container!')"
