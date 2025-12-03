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

export SCENE_LABELS=(
camera_lights
no_lights
velocity
)

for scene_name in ${SCENE_NAMES[*]}
do
    for scene_label in ${SCENE_LABELS[*]}
    do
        printf "Evaluating label/scene:   %s\n" "${scene_label}/${scene_name}"

        python /scripts/eval_semseg.py \
            --approach 'openscene' \
            --semantic_info_path "/data/datasets/generated/hm3d/${scene_label}/${scene_name}/embed_semseg_classes.json" \
            --scene_label_set \
            --excluded_classes "0" \
            --pred_pc_path "/home/docker_user/OpenScene/output/hm3d/results/${scene_label}/${scene_name}/fusion" \
            --gt_pc_path "/data/gt/generated/hm3d/no_lights/${scene_name}/pointcloud.pcd" \
            --output_path "/results/osma-bench/openscene/hm3d/${scene_label}" \
            --result_tag "${scene_name}" \
            --clip_prompts "a {} in a scene" \
            --clip_name "ViT-L-14-336-quickgelu" \
            --clip_pretrained "openai" \
            --nn_count 1
    done
done

for scene_label in ${SCENE_LABELS[*]}
do
    printf "Metrics for label:   %s\n" "${scene_label}"

    python /scripts/compute_metrics.py \
        --results_dir "/results/osma-bench/openscene/hm3d/${scene_label}" \
        --excluded "-1 0"
done