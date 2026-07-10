#!/bin/bash
#SBATCH --job-name=prod_rf
#SBATCH --partition=sequana_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --chdir=/scratch/pcmrnbio2/alex.yumbo/combinations_predictions
#SBATCH --output=/scratch/pcmrnbio2/alex.yumbo/logs/prod_rf_%j.out
#SBATCH --error=/scratch/pcmrnbio2/alex.yumbo/logs/prod_rf_%j.err

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

CONTAINER_IMG="/scratch/pcmrnbio2/alex.yumbo/containers/py_ml.sif"
REPO_DIR="/scratch/pcmrnbio2/alex.yumbo/combinations_predictions"

singularity exec -B $REPO_DIR:/app $CONTAINER_IMG \
    python -u -m src.tuning_ml_models \
        --strain_embs "data/strains_embs/*.npz" \
        --splits_path "data/strains_embs/train_test_splits_indices.npy" \
        --output_dir "data/models_production/" \
        --n_iter 30 \
        --cv 5 \
        --model rf \
        --n_jobs $SLURM_CPUS_PER_TASK