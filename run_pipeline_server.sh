#!/bin/bash
#SBATCH --job-name=embs_generation
#SBATCH --nodes=1
#SBATCH --partition=sequana_gpu_dev       # Obrigatoriamente sequana_gpu ou gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Solicita 1 GPU para o processo
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/pcmrnbio2/alex.yumbo/logs/job_%j.out
#SBATCH --error=/scratch/pcmrnbio2/alex.yumbo/logs/job_%j.err

SCRATCH_DIR="/scratch/pcmrnbio2/$(whoami)"
REPO_DIR="/prj/pcmrnbio2/alex.yumbo/combinations_predictions"
# Path to containers 
CONTAINER_DATA="$SCRATCH_DIR/containers/py_data.sif"
CONTAINER_EMBS="$SCRATCH_DIR/containers/py_embs.sif"
CONTAINER_ML="$SCRATCH_DIR/containers/py_ml.sif"


cd $SLURM_SUBMIT_DIR
echo "Directory: $SLURM_SUBMIT_DIR"

# 
echo "Test if singularity is working..."


singularity exec \
    --bind $REPO_DIR:/app \
    --bind $SCRATCH_DIR:$SCRATCH_DIR \
    $CONTAINER_EMBS \
    python --version

# singularity exec \
#     --bind $REPO_DIR:/app \
#     --bind $SCRATCH_DIR:$SCRATCH_DIR \
#     $CONTAINER_EMBS \
#     python src/smiles2clsembs.py \
#         --smiles_path $SCRATCH_DIR/data/normalized_abx.csv \
#         --smiles_column normalized_smiles \
#         --model_name DeepChem/ChemBERTa-77M-MLM \
#         --output_path $SCRATCH_DIR/data/abx_embs.npz

# echo "Generating embeddings for small molecules..."

# singularity exec \
#     --bind $REPO_DIR:/app \
#     --bind $SCRATCH_DIR:$SCRATCH_DIR \
#     $CONTAINER_EMBS \
#     python src/smiles2clsembs.py \
#         --smiles_path $SCRATCH_DIR/data/normalized_small_mol.csv \
#         --smiles_column normalized_smiles \
#         --model_name DeepChem/ChemBERTa-77M-MLM \
#         --output_path $SCRATCH_DIR/data/small_mol_embs.npz

