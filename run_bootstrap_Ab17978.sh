#!/bin/bash
#SBATCH --job-name=boot_Ab17978
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --chdir=/scratch/pcmrnbio2/alex.yumbo/combinations_predictions
#SBATCH --output=/scratch/pcmrnbio2/alex.yumbo/logs/boot_dev_Ab17978_%j.out
#SBATCH --error=/scratch/pcmrnbio2/alex.yumbo/logs/boot_dev_Ab17978_%j.err

# Define all 6 strains to process
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
        --OUTPUT_DIR "data/bootstrap_results_test/" \
        --STRAIN "$STRAIN" \
        --MODEL_NAME "$MODEL" \
        --N_BOOTSTRAPS 2