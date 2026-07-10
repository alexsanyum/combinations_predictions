#!/bin/bash
#SBATCH --job-name=concat_embs
#SBATCH --partition=sequana_cpu_dev       
#SBATCH --nodes=1                           # We want 1 physical node
#SBATCH --ntasks=1                          # 1 single Python execution path
#SBATCH --cpus-per-task=44                  # 22 physical cores x 2 threads (1 full socket)
#SBATCH --hint=nomultithread                # Optional: Optimizes for physical cores if preferred
#SBATCH --mem=180G                          # Allocates plenty of RAM on that socket's channel
#SBATCH --time=00:20:00                     # Max allowed development time
#SBATCH --chdir=/scratch/pcmrnbio2/alex.yumbo/combinations_predictions
#SBATCH --output=/scratch/pcmrnbio2/alex.yumbo/logs/job_%j.out
#SBATCH --error=/scratch/pcmrnbio2/alex.yumbo/logs/job_%j.err

# --- Environment setup for maximum RAM/CPU efficiency ---
REPO_DIR="/scratch/pcmrnbio2/alex.yumbo/combinations_predictions"
CONTAINER_IMG="/scratch/pcmrnbio2/alex.yumbo/containers/py_data.sif"

# Force internal math libraries to use all 44 allocated threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting highly optimized run..."

singularity exec \
    -B $REPO_DIR:/app \
    $CONTAINER_IMG \
    python -u /app/src/concat_strain_embs.py \
        --comb_data /app/data/valid_comb_data.csv \
        --norm_small_mols /app/data/normalized_small_mol.csv \
        --small_mol_embs /app/data/small_mol_embs.npz \
        --norm_abx /app/data/normalized_abx.csv \
        --abx_embs /app/data/abx_embs.npz \
        --output_dir /app/data/strains_embs/