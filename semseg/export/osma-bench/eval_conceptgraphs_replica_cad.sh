#!/bin/bash

export DATASET_ROOT=/data/datasets/generated/replica_cad

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
export CLASS_SET=none
export THRESHOLD=1.2

for scene_name in ${SCENE_NAMES[*]}
do
    for scene_label in ${SCENE_LABELS[*]}
    do
        printf "Evaluating label/scene:   %s\n" "${scene_label}/${scene_name}"

        python /scripts/eval_semseg.py \
            --approach 'conceptgraphs' \
            --semantic_info_path "${DATASET_ROOT}/${scene_label}/${scene_name}/embed_semseg_classes.json" \
            --scene_label_set \
            --excluded_classes "0" \
            --pred_pc_path "${DATASET_ROOT}/${scene_label}/${scene_name}/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz" \
            --gt_pc_path "/data/gt/generated/replica_cad/${scene_name}/semantic.pcd" \
            --output_path "/results/osma-bench/conceptgraphs/replica_cad/${scene_label}" \
            --result_tag "${scene_name}" \
            --clip_prompts "an image of {}" \
            --clip_name "ViT-H-14" \
            --clip_pretrained "laion2b_s32b_b79k" \
            --nn_count 1
    done
done

for scene_label in ${SCENE_LABELS[*]}
do
    printf "Metrics for label:   %s\n" "${scene_label}"
    
    python /scripts/compute_metrics.py \
        --results_dir "/results/osma-bench/conceptgraphs/replica_cad/${scene_label}" \
        --excluded "-1 0"
done