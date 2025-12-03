#!/bin/bash

export OPENSEG_PATH=/assets/openseg_exported_clip
export FEATURE_DIR_PREFIX=/home/docker_user/OpenScene/output/hm3d/fusion
export TMP_DIR_PATH=/home/docker_user/OpenScene/output/hm3d
export DATASET_DIR_PATH=/data/datasets/generated/hm3d
export RESULTS_PATH=/home/docker_user/OpenScene/output/hm3d/results
export CONFIG_ROOT=/home/docker_user/OpenScene/config/replica
export STRIDE=30

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

export SCENE_LABELS=(
    no_lights
    velocity
    camera_lights
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

        sh ./run/eval.sh $RESULTS_PATH/$scene_label $CONFIG_ROOT/hm3d_openseg_pretrained.yaml fusion 
    done
done
