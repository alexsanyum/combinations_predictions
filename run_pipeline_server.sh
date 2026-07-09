#!/bin/bash
#SBATCH --job-name=embs_generation
#SBATCH --nodes=1
#SBATCH --partition=sequana_gpu_dev       # Obrigatoriamente sequana_gpu ou gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Solicita 1 GPU para o processo
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/pcmrnbio2/alex.yumbo/logs/job_%j.out
#SBATCH --error=/scratch/pcmrnbio2/alex.yumbo/logs/job_%j.err


REPO_DIR="/scratch/pcmrnbio2/alex.yumbo/combinations_predictions"
CONTAINER_IMG="/scratch/pcmrnbio2/alex.yumbo/containers/py_embs.sif"

singularity exec --nv \
    -B $REPO_DIR:/app \
    $CONTAINER_IMG \
    python /app/src/smiles2clsembs.py --smiles_path /app/data/normalized_abx.csv \
    --smiles_column normalized_smiles \
    --model_name DeepChem/ChemBERTa-77M-MLM \
    --output_path /app/data/abx_embs.npz
