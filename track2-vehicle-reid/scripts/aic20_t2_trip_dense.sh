#!/usr/bin/env bash
source "scripts/master_env.sh"
exp_id="aic20_t2_trip_dense"
python main.py \
    --gpu_id $GPUID \
    -w $N_WORKERS \
    --dataset_cfg "./configs/dataset_cfgs/aic20_vehicle_reid.yaml" \
    --model_cfg   "./configs/model_cfgs/${exp_id}.yaml" \
    --train_cfg   "./configs/train_cfgs/${exp_id}.yaml" \
    --logdir      "logs/${exp_id}" \
    --log_fname   "logs/${exp_id}/stdout.log" \
    --train_mode  "from_scratch" \
    --is_training true \