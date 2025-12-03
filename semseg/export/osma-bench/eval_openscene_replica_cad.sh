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

export SCENE_LABELS=(baseline camera_lights dynamic_lights no_lights velocity)

for scene_name in ${SCENE_NAMES[*]}
do
    for scene_label in ${SCENE_LABELS[*]}
    do
        printf "Evaluating label/scene:   %s\n" "${scene_label}/${scene_name}"

        python /scripts/eval_semseg.py \
            --approach 'openscene' \
            --semantic_info_path "/data/datasets/generated/replica_cad/${scene_label}/${scene_name}/embed_semseg_classes.json" \
            --scene_label_set \
            --excluded_classes "0" \
            --pred_pc_path "/home/docker_user/OpenScene/output/replica_cad/results/${scene_label}/${scene_name}/fusion" \
            --gt_pc_path "/data/gt/generated/replica_cad/${scene_name}/pointcloud.pcd" \
            --output_path "/results/osma-bench/openscene/replica_cad/${scene_label}" \
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
        --results_dir "/results/osma-bench/openscene/replica_cad/${scene_label}" \
        --excluded "-1 0"
done