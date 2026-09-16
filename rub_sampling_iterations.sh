#!/bin/bash
#SBATCH --job-name=iter_undersampling
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --mem=64G
#SBATCH --chdir=/scratch/pcmrnbio2/alex.yumbo/combinations_predictions
#SBATCH --output=/scratch/pcmrnbio2/alex.yumbo/logs/iter_undersampling_%j.out
#SBATCH --error=/scratch/pcmrnbio2/alex.yumbo/logs/iter_undersampling_%j.err

PATH_TO_DATA=$1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

CONTAINER_IMG="/scratch/pcmrnbio2/alex.yumbo/containers/py_ml.sif"
REPO_DIR="/scratch/pcmrnbio2/alex.yumbo/combinations_predictions"

singularity exec -B $REPO_DIR:/app $CONTAINER_IMG \
        python -u -m src.undersampling_iterations \
        --data_path "$PATH_TO_DATA" \
        --output_dir "data/undersampling_results/" \
        --n_jobs 32 \
        --iterations 3 \
        