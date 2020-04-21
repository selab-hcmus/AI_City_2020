#!/usr/bin/env bash
source "scripts/master_env.sh"

python main.py \
    --gpu_id $GPUID \
    --dataset_cfg "./configs/dataset_cfgs/aic20_vehicle_type.yaml" \
    --model_cfg   "./configs/model_cfgs/aic20_vehicle_type.yaml" \
    --train_cfg   "./configs/train_cfgs/aic20_vehicle_type.yaml" \
    --logdir      "logs/aic20_vehicle_type" \
    --log_fname   "logs/aic20_vehicle_type/stdout.log" \
    --is_training true \
    --train_mode  "from_scratch" \
    --is_data_augmented true \
    -w $N_WORKERS
