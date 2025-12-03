#!/bin/bash

export SCENE_NAMES=(
00800-TEEsavR23oF
00802-wcojb4TFT35
00803-k1cupFYWXJ6
00808-y9hTuugGdiq
00810-CrMo8WxCyVb
00813-svBbv1Pavdk
00814-p53SfW6mjZe
00815-h1zeeAwLh9Z
)

for scene_name in ${SCENE_NAMES[*]}
do
    printf "Reconstructing scene:   %s\n" "${scene_name}"

    python /scripts/run_slam.py \
        --dataset_root "/data/datasets/generated/hm3d/no_lights/" \
        --output_dir "/data/gt/generated/hm3d/no_lights" \
        --scene_id ${scene_name} \
        --stride 5 \
        --downsample_rate 10 \
        --load_semseg
done