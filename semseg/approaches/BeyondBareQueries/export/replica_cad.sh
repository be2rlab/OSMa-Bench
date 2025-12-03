#!/bin/bash

export SCENE_NAMES=(
apt_0
)
# apt_0
# apt_3
# v3_sc0_staging_00
# v3_sc0_staging_12
# v3_sc0_staging_16
# v3_sc0_staging_19
# v3_sc0_staging_20
# v3_sc1_staging_00
# v3_sc1_staging_06
# v3_sc1_staging_12
# v3_sc1_staging_19
# v3_sc1_staging_20
# v3_sc2_staging_00
# v3_sc2_staging_11
# v3_sc2_staging_13
# v3_sc2_staging_19
# v3_sc2_staging_20
# v3_sc3_staging_03
# v3_sc3_staging_04
# v3_sc3_staging_08
# v3_sc3_staging_15
# v3_sc3_staging_20

export SCENE_LABELS=(baseline) # baseline camera_lights dynamic_lights no_lights velocity

for SCENE_NAME in ${SCENE_NAMES[*]}
do
    for SCENE_LABEL in ${SCENE_LABELS[*]}
    do
        printf "Running scene:   %s\n" "$SCENE_NAME/${SCENE_LABEL}"

        python main.py --config_path "examples/configs/replica_cad/${SCENE_LABEL}_${SCENE_NAME}.yaml" --save_path="export/output/"
    	# python3 visualize/show_construction.py --animation_folder="export/output"
    done
done
