#!/bin/bash

export BBQ_DIR=/home/docker_user/BeyondBareQueries

export SCENE_NAMES=(
apt_0
apt_3
v3_sc0_staging_00
v3_sc0_staging_12
v3_sc0_staging_16
v3_sc0_staging_19
v3_sc0_staging_20
v3_sc1_staging_00
v3_sc1_staging_06
v3_sc1_staging_12
v3_sc1_staging_19
v3_sc1_staging_20
v3_sc2_staging_00
v3_sc2_staging_11
v3_sc2_staging_13
v3_sc2_staging_19
v3_sc2_staging_20
v3_sc3_staging_03
v3_sc3_staging_04
v3_sc3_staging_08
v3_sc3_staging_15
v3_sc3_staging_20
)

export SCENE_LABELS=(baseline camera_lights dynamic_lights no_lights velocity)


for scene_name in ${SCENE_NAMES[*]}
do
    for scene_label in ${SCENE_LABELS[*]}
    do
        printf "Running label/scene:   %s\n" "${scene_label}/${scene_name}"

        python3 ~/BeyondBareQueries/main.py \
            --config_path /home/docker_user/BeyondBareQueries/examples/configs/replica_cad/${scene_label}_${scene_name}.yaml

        # python3 ~/BeyondBareQueries/main.py \
        #     --config_path "${BBQ_DIR}/examples/configs/replica_cad/${scene_label}_${scene_name}.yaml" \
        #     --save_path "${BBQ_DIR}/output/visuals/replica_cad/${scene_label}/${scene_name}/"

        # python3 visualize/show_construction.py \
        #     --animation_folder "${BBQ_DIR}/output/visuals/replica_cad/${scene_label}/${scene_name}/"
    done
done