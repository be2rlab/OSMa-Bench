#!/bin/bash

export OPENSEG_PATH=/assets/openseg_exported_clip
export FEATURE_DIR_PREFIX=/home/docker_user/OpenScene/output/replica_cad/fusion
export TMP_DIR_PATH=/home/docker_user/OpenScene/output/replica_cad
export DATASET_DIR_PATH=/data/datasets/generated/replica_cad/
export RESULTS_PATH=/home/docker_user/OpenScene/output/replica_cad/results
export CONFIG_ROOT=/home/docker_user/OpenScene/config/replica
export STRIDE=30

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

export SCENE_LABELS=(
    baseline 
    camera_lights
    dynamic_lights
    no_lights
    velocity
)

cd /home/docker_user/OpenScene

for scene_name in ${SCENE_NAMES[*]}
do 
    for scene_label in ${SCENE_LABELS[*]}
    do
        printf "Running scene:   %s\n" "$scene_label/${scene_name}"

        export CONFIG_NAME=${scene_label}
        export SCENE_NAME=${scene_name}

        python /scripts/run_slam.py \
            --dataset_root "${DATASET_DIR_PATH}/${scene_label}/" \
            --scene_id ${scene_name} \
            --stride $STRIDE \
            --downsample_rate 15 \
            --load_semseg \
            --output_dir $TMP_DIR_PATH/rgb_clouds/$scene_label

        python3 ./scripts/preprocess/preprocess_replica.py \
            $DATASET_DIR_PATH/$scene_label/$scene_name \
            $TMP_DIR_PATH/rgb_clouds/$scene_label/$scene_name \
            $TMP_DIR_PATH \
            $STRIDE 

        python ./scripts/feature_fusion/replica_openseg.py \
            --data_dir $TMP_DIR_PATH \
            --scene $scene_name \
            --config $scene_label \
            --output_dir $FEATURE_DIR_PREFIX/$scene_label \
            --split test \
            --openseg_model $OPENSEG_PATH

        sh ./run/eval.sh $RESULTS_PATH/$scene_label $CONFIG_ROOT/replica_openseg_pretrained.yaml fusion 
    done
done
