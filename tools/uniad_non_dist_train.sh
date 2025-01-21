#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
GPUS=$2                                              #
# -------------------------------------------------- #
# Determine the number of GPUs to use per node
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

# Set default values for MASTER_PORT, MASTER_ADDR, and RANK
MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}

# Set the working directory
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

# Create logs directory if it doesn't exist
if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

# Run training script without distributed training
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train.py \
    $CFG \
    --launcher none \
    --deterministic \
    --work-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/train.$T


#--resume-from projects/work_dirs/stage3_e2e_IMU/base_e2e_IMU/latest.pth \
#那个total_epochs = 40   #注意，每训20个epoch，想再继续的话这这里就得改大！也要改