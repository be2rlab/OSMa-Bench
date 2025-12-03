#!/bin/bash

export SCENE_NAMES=(00800-TEEsavR23oF) #  00802-wcojb4TFT35 00803-k1cupFYWXJ6 00808-y9hTuugGdiq  
export SCENE_LABELS=(camera_lights_1) # no_lights_1 velocity_1 camera_lights_2 no_lights_2 velocity_2 

for SCENE_NAME in ${SCENE_NAMES[*]}
do
    for SCENE_LABEL in ${SCENE_LABELS[*]}
    do
        printf "Running scene:   %s\n" "$SCENE_NAME/${SCENE_LABEL}"

        python main.py --config_path "examples/configs/hm3d/${SCENE_LABEL}_${SCENE_NAME}.yaml"
    done
done


# for SCENE_NAME in ${SCENE_NAMES[*]}
# do
#     for SCENE_LABEL in ${SCENE_LABELS[*]}
#     do
#         printf "Evaluating label/scene:   %s\n" "${SCENE_LABEL}/${SCENE_NAME}"

#         ######
#         # Then run the following commands to evaluate the semantic segmentation results.
#         ######
#         if [[ "$SCENE_LABEL" == *"_1" ]]; then
#             export SEMANTIC_SCENE_LABEL="no_lights_1"
#         else
#             export SEMANTIC_SCENE_LABEL="no_lights_2"
#         fi

#         python /scripts/eval_semseg.py \
#             --replica_root "${DATASET_ROOT}/${SCENE_LABEL}" \
#             --replica_semantic_root "${DATASET_ROOT}/${SEMANTIC_SCENE_LABEL}" \
#             --n_exclude 1 \
#             --label "${SCENE_LABEL}" \
#             --scene_ids_str "${SCENE_NAME}" \
#             --tag "${SCENE_NAME}_${SCENE_LABEL}" \
#             --pred_exp_name "none_overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub" \
#             --results_path "/results/conceptgraphs/hm3d_minival/" \
#             --semseg_classes "${DATASET_ROOT}/${SCENE_LABEL}/${SCENE_NAME}/embed_semseg_classes.json" # \
#             # --device "cpu"
#     done
# done
