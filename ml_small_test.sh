#!/bin/bash
#SBATCH --job-name=ml_small_test
#SBATCH --partition=sequana_cpu_dev         # Use the development partition for quick tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                   # 4 cores is plenty for this small scale
#SBATCH --mem=64G                           # Minimal RAM needed for 1000 rows
#SBATCH --time=00:20:00                     # Max allowed dev partition time
#SBATCH --chdir=/scratch/pcmrnbio2/alex.yumbo/combinations_predictions
#SBATCH --output=/scratch/pcmrnbio2/alex.yumbo/logs/small_test_%j.out
#SBATCH --error=/scratch/pcmrnbio2/alex.yumbo/logs/small_test_%j.err

# --- Environment Setup ---
CONTAINER_IMG="/scratch/pcmrnbio2/alex.yumbo/containers/py_ml.sif"
REPO_DIR="/scratch/pcmrnbio2/alex.yumbo/combinations_predictions"

# Bind internal math threading to our allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# List of all models to evaluate sequentially
MODELS=("lgbm" "xgb" "rf" "lr" "svc")

echo "=========================================================="
echo "STARTING ALL-MODEL SANITY TEST (n_iter=2, cv=2)"
echo "=========================================================="

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "----------------------------------------------------------"
    echo "LAUNCHING MODEL: $MODEL"
    echo "----------------------------------------------------------"
    
    # Executing using the module style (-m) as specified in your test line
    singularity exec -B $REPO_DIR:/app $CONTAINER_IMG \
        python -u -m src.tuning_ml_models \
            --strain_embs "data/strains_embs/*.npz" \
            --splits_path "data/strains_embs/train_test_splits_indices.npy" \
            --output_dir "data/models_small_test/" \
            --n_iter 2 \
            --cv 2 \
            --model "$MODEL" \
            --n_jobs 1
            
    echo "FINISHED MODEL: $MODEL (Exit Code: $?)"
done

echo "=========================================================="
echo "ALL MODEL TESTS COMPLETED."
echo "=========================================================="