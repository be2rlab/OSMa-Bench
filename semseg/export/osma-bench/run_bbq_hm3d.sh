#!/bin/bash

export BBQ_DIR=/home/docker_user/BeyondBareQueries

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

export SCENE_LABELS=(camera_lights no_lights velocity)

for scene_name in ${SCENE_NAMES[*]}
do
    for scene_label in ${SCENE_LABELS[*]}
    do
        printf "Running label/scene:   %s\n" "${scene_label}/${scene_name}"

        python3 ~/BeyondBareQueries/main.py \
            --config_path "${BBQ_DIR}/examples/configs/hm3d/${scene_label}_${scene_name}.yaml"

        # python3 ~/BeyondBareQueries/main.py \
        #     --config_path "${BBQ_DIR}/examples/configs/hm3d/${scene_label}_${scene_name}.yaml" \
        #     --save_path "${BBQ_DIR}/output/visuals/hm3d/${scene_label}/${scene_name}/"

        # python3 visualize/show_construction.py \
        #     --animation_folder "${BBQ_DIR}/output/visuals/hm3d/${scene_label}/${scene_name}/"
    done
done