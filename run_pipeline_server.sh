#!/bin/bash

SCRATCH_DIR="/scratch/pcmrnbio2/$(whoami)"
CONTAINER_DATA="$SCRATCH_DIR/containers/py_data.sif"
REPO_DIR="/prj/pcmrnbio2/alex.yumbo/combinations_predictions"

cd $REPO_DIR
pwd

singularity exec \
    --bind $REPO_DIR:/app \
    --bind $SCRATCH_DIR:$SCRATCH_DIR \
    $CONTAINER_DATA \
    python src/process_valid_SMILES_comb.py \
        --comb_data_path $SCRATCH_DIR/data/B_blisssum_DropArray.csv \
        --abx_data_path $REPO_DIR/metadata/antibiotics_names.txt \
        --output_path $SCRATCH_DIR/data/


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