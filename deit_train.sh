#!/bin/bash

NUM_PROC=4
MODEL=deit_small_patch16_224
BATCH_SIZE=256
DATA_PATH=/data/ILSVRC2012
EPOCH=90
LR=2.5e-4


DIR=/home/shkim/EfML
VERSION=Deit_HLA_90_ADA6000
OUTPUT_DIR=${DIR}/${VERSION}
LOGFILE=${OUTPUT_DIR}.log
mkdir -p ${OUTPUT_DIR}
cd ${DIR} 
export VERSION

# 실행
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=$NUM_PROC \
    --use_env \
    ./main.py \
    --model $MODEL \
    --no-model-ema \
    --batch-size $BATCH_SIZE \
    --data-path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCH \
    --lr $LR 
    # >> "${LOGFILE}" 2>&1

