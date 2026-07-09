#!/bin/bash

SCRATCH_DIR="/scratch/pcmrnbio2/$(whoami)"
DATA_PATH="$SCRATCH_DIR/data/"
CONTAINER_DATA="$SCRATCH_DIR/containers/py_data.sif"
REPO_DIR="/prj/pcmrnbio2/alex.yumbo/combinations_predictions"

# Print path for debugging
echo "Scratch directory: $SCRATCH_DIR"
echo "Contetnt of scratch directory:"
ls "$SCRATCH_DIR"

echo "Data path: $DATA_PATH"
echo "Contents of data directory:"
ls "$DATA_PATH"
echo "Container data path: $CONTAINER_DATA"
echo "Repository directory: $REPO_DIR"


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