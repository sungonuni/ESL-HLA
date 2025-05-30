python simclr_main.py \
    --GPU_USE 0 \
    --RUN_NAME simclr_HLS \
    --DEBUG_MODE False \
    --MODEL Q_simclr \
    --DATASET cifar10 \
    --AMP false \
    --EPOCHS 200 \
    --BATCH_SIZE 256 \
    --LR 0.0003 \
    --SEED 2025 \
    --quantBWDGogi no \
    --quantBWDWgt no \
    --quantBWDGogw no \
    --quantBWDAct no \
    \
    --transform_scheme gih_gwlr \
    --TransformGogi true \
    --TransformWgt true \
    --TransformGogw true \
    --TransformInput true

python simclr_valid.py \
    --GPU_USE 0 \
    --RUN_NAME simclr_HLS_FP \
    --DEBUG_MODE true \
    --MODEL simclr \
    --DATASET cifar10 \
    --AMP false \
    --EPOCHS 200 \
    --BATCH_SIZE 256 \
    --LR 0.0003 \
    --SEED 2025