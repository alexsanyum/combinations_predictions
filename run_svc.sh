#!/bin/bash
#SBATCH --job-name=prod_svc
#SBATCH --partition=sequana_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --chdir=/scratch/pcmrnbio2/alex.yumbo/combinations_predictions
#SBATCH --output=/scratch/pcmrnbio2/alex.yumbo/logs/prod_svc_%j.out
#SBATCH --error=/scratch/pcmrnbio2/alex.yumbo/logs/prod_svc_%j.err

# Allow internal solvers to multithread across all 16 cores natively
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

CONTAINER_IMG="/scratch/pcmrnbio2/alex.yumbo/containers/py_ml.sif"
REPO_DIR="/scratch/pcmrnbio2/alex.yumbo/combinations_predictions"

singularity exec -B $REPO_DIR:/app $CONTAINER_IMG \
    python -u -m src.tuning_ml_models \
        --strain_embs "data/strains_embs/*.npz" \
        --splits_path "data/strains_embs/train_test_splits_indices.npy" \
        --output_dir "data/models_production/" \
        --n_iter 30 \
        --cv 5 \
        --model svc \
        --n_jobs 1