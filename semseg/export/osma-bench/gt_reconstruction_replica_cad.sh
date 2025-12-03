#!/bin/bash

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

for scene_name in ${SCENE_NAMES[*]}
do
    printf "Reconstructing scene:   %s\n" "${scene_name}"

    python /scripts/run_slam.py \
        --dataset_root "/data/datasets/generated/replica_cad/baseline/" \
        --output_dir "/data/gt/generated/replica_cad/" \
        --scene_id ${scene_name} \
        --stride 5 \
        --downsample_rate 10 \
        --load_semseg
done