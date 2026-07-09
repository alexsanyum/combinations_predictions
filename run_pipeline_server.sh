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

echo "Current working directory: $(pwd)"
cd $REPO_DIR
echo "Changed working directory to: $(pwd)"

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


#      sequana_cpu_dev       1         1        4      192    00:20:00 
#   sequana_cpu_shared       0         0        0        0    00:00:00 
#      sequana_gpu_dev       1         1        4      192    00:20:00 
#   sequana_gpu_shared       0         0        0        0    00:00:00 
#          sequana_cpu       4        24       50     2400  4-00:00:00 
#     sequana_cpu_long       3        18       10      480 31-00:00:00 
#   sequana_cpu_bigmem       4        24       18      846  4-00:00:00 
# sequana_cpu_bigmem_+       3        18        5      240 31-00:00:00 
#          sequana_gpu       4        24       24     1152  4-00:00:00 
#     sequana_gpu_long       3        18       10      480 31-00:00:00 