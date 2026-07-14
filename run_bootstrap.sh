#!/bin/bash
#SBATCH --job-name=bootstrap
#SBATCH --partition=sequana_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --mem=120G
#SBATCH --chdir=/scratch/pcmrnbio2/alex.yumbo/combinations_predictions
#SBATCH --output=/scratch/pcmrnbio2/alex.yumbo/logs/bootstrap_%j.out
#SBATCH --error=/scratch/pcmrnbio2/alex.yumbo/logs/bootstrap_%j.err

# ------------------------------------------------------------------
# 1. Capture the Strain name passed from the command line ($1)
# ------------------------------------------------------------------
STRAIN=$1

# Quick safety check: Ensure the user provided a strain
if [ -z "$STRAIN" ]; then
    echo "ERROR: No strain specified. Usage: sbatch run_bootstrap.sh <STRAIN_NAME>"
    exit 1
fi
MODEL="lgbm" # Modify this if you are running other architectures!

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK


CONTAINER_IMG="/scratch/pcmrnbio2/alex.yumbo/containers/py_ml.sif"
REPO_DIR="/scratch/pcmrnbio2/alex.yumbo/combinations_predictions"

singularity exec -B $REPO_DIR:/app $CONTAINER_IMG \
        python -u -m src.bootstrap_analysis \
        --PATH_TO_MODELS "data/models_production/" \
        --PATH_TO_DATA "data/strains_embs/" \
        --PATH_TO_INDICES "data/strains_embs/train_test_splits_indices.npy" \
        --OUTPUT_DIR "data/bootstrap_results/" \
        --STRAIN "$STRAIN" \
        --MODEL_NAME "$MODEL" \
        --N_BOOTSTRAPS 1000