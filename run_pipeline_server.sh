#!/bin/bash
#SBATCH --job-name=embs_generation
#SBATCH --nodes=1
#SBATCH --partition=sequana_gpu       
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  
#SBATCH --time=00:10:00
#SBATCH --chdir=/scratch/pcmrnbio2/alex.yumbo/combinations_predictions
#SBATCH --output=/scratch/pcmrnbio2/alex.yumbo/logs/job_%j.out
#SBATCH --error=/scratch/pcmrnbio2/alex.yumbo/logs/job_%j.err

REPO_DIR="/scratch/pcmrnbio2/alex.yumbo/combinations_predictions"
CONTAINER_IMG="/scratch/pcmrnbio2/alex.yumbo/containers/py_embs.sif"

# Dedicated cache for HuggingFace models
HF_CACHE_DIR="/scratch/pcmrnbio2/alex.yumbo/huggingface_cache"
mkdir -p $HF_CACHE_DIR 

# Create a fake home space on scratch so the container can write internal logs safely
FAKE_HOME="/scratch/pcmrnbio2/alex.yumbo/fake_home"
mkdir -p $FAKE_HOME

# Export variables for HuggingFace
export TRANSFORMERS_CACHE=$HF_CACHE_DIR
export HF_HOME=$HF_CACHE_DIR

# First run to download the model and cache it
singularity exec --nv \
    --home $FAKE_HOME \
    --writable-tmpfs \
    -B $REPO_DIR:/app \
    -B $HF_CACHE_DIR:$HF_CACHE_DIR \
    $CONTAINER_IMG \
    python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM'); \
    AutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MLM')"

echo "Model downloaded and cached successfully."

# Run Singularity with the --home redirect and --writable-tmpfs layer
singularity exec --nv \
    --home $FAKE_HOME \
    --writable-tmpfs \
    -B $REPO_DIR:/app \
    -B $HF_CACHE_DIR:$HF_CACHE_DIR \
    $CONTAINER_IMG \
    python /app/src/smiles2clsembs.py --smiles_path /app/data/normalized_abx.csv \
    --smiles_column normalized_smiles \
    --model_name DeepChem/ChemBERTa-77M-MLM \
    --output_path /app/data/abx_embs.npz