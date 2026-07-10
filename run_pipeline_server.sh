#!/bin/bash
#SBATCH --job-name=concat_embs
#SBATCH --nodes=1
#SBATCH --partition=sequana_cpu_dev       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8               
#SBATCH --time=00:10:00
#SBATCH --mem=64G
#SBATCH --chdir=/scratch/pcmrnbio2/alex.yumbo/combinations_predictions
#SBATCH --output=/scratch/pcmrnbio2/alex.yumbo/logs/job_%j.out
#SBATCH --error=/scratch/pcmrnbio2/alex.yumbo/logs/job_%j.err

REPO_DIR="/scratch/pcmrnbio2/alex.yumbo/combinations_predictions"
CONTAINER_IMG="/scratch/pcmrnbio2/alex.yumbo/containers/py_data.sif"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Test if is accessing to the repository..."

singularity exec \
    -B $REPO_DIR:/app \
    $CONTAINER_IMG \
    python /app/src/concat_strain_embs.py --comb_data /app/data/valid_comb_data.csv \
                                      --norm_small_mols /app/data/normalized_small_mol.csv \
                                      --small_mol_embs /app/data/small_mol_embs.npz \
                                      --norm_abx /app/data/normalized_abx.csv \
                                      --abx_embs /app/data/abx_embs.npz \
                                      --output_dir /app/data/strains_embs/
