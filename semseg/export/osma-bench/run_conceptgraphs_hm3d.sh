#!/bin/bash

export GSA_PATH=/home/docker_user/Grounded-Segment-Anything
export ASSETS_PATH=/home/docker_user/ConceptGraphs/checkpoints

export DATASET_ROOT=/data/datasets/generated/hm3d

export CG_FOLDER=/home/docker_user/ConceptGraphs/conceptgraph
export DATASET_CONFIG_PATH=${CG_FOLDER}/dataset/dataconfigs/replica/hm3d.yaml

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

export CLASS_SET=none
export THRESHOLD=1.2

mkdir -p "/tmp/config"

for scene_name in ${SCENE_NAMES[*]}
do
    for scene_label in ${SCENE_LABELS[*]}
    do
        printf "Running scene:   %s\n" "$scene_name/${scene_label}"

        cp ${DATASET_CONFIG_PATH} "/tmp/config/data_config.yaml"
        echo "" >> "/tmp/config/data_config.yaml"
        cat "${DATASET_ROOT}/${scene_label}/${scene_name}/camera_params.yaml" >> "/tmp/config/data_config.yaml"

        ######
        # ConceptGraphs uses SAM in the "segment all" mode and extract class-agnostic masks.
        ######
        python ${CG_FOLDER}/scripts/generate_gsa_results.py \
            --dataset_root ${DATASET_ROOT} \
            --dataset_config "/tmp/config/data_config.yaml" \
            --scene_id "${scene_label}/${scene_name}/" \
            --class_set ${CLASS_SET} \
            --stride 5

        ######
        # The following command builds an object-based 3D map of the scene, using the image segmentation results from above.
        ######
        python ${CG_FOLDER}/slam/cfslam_pipeline_batch.py \
            dataset_root=${DATASET_ROOT} \
            dataset_config="/tmp/config/data_config.yaml" \
            stride=5 \
            scene_id="${scene_label}/${scene_name}/" \
            spatial_sim_type=overlap \
            mask_conf_threshold=0.95 \
            match_method=sim_sum \
            sim_threshold=${THRESHOLD} \
            dbscan_eps=0.1 \
            gsa_variant=none \
            class_agnostic=True \
            skip_bg=True \
            max_bbox_area_ratio=0.5 \
            save_suffix=overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub \
            merge_interval=20 \
            merge_visual_sim_thresh=0.8 \
            merge_text_sim_thresh=0.8 \
            save_objects_all_frames=True

        # ######
        # # If save_objects_all_frames=True was used to save the mapping results at every frame, which can be used for animated visualization by:
        # ######
        # python ${CG_FOLDER}/scripts/animate_mapping_interactive.py \
        #     --input_folder ${DATASET_ROOT}/${scene_label}/${scene_name}/objects_all_frames/none_overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub

        # ######
        # # If save_objects_all_frames=True was used to save the mapping results at every frame, which can be used for animated visualization by:
        # ######
        # python ${CG_FOLDER}/scripts/animate_mapping_save.py \
        #     --input_folder ${DATASET_ROOT}/${scene_label}/${scene_name}/objects_all_frames/none_overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub

        # ######
        # # Visualize the object-based mapping results. You can use keys b, c, r, f, i.
        # ######
        # python ${CG_FOLDER}/scripts/visualize_cfslam_results.py \
        #     --result_path ${DATASET_ROOT}/${scene_label}/${scene_name}/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub.pkl.gz
    done
done

######
# Generate a scene graph
######

# export NEW_ROOT="${DATASET_ROOT}/../cg_hm3d"
# mkdir -p "${NEW_ROOT}"

for scene_name in ${SCENE_NAMES[*]}
do
    for scene_label in ${SCENE_LABELS[*]}
    do
        printf "Graph building label/scene:   %s\n" "${scene_label}/${scene_name}"

        PKL_FILENAME=full_pcd_none_overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz

        python ${CG_FOLDER}/scenegraph/build_scenegraph_cfslam.py \
            --mode extract-node-captions \
            --cachedir ${DATASET_ROOT}/${scene_label}/${scene_name}/sg_cache \
            --mapfile ${DATASET_ROOT}/${scene_label}/${scene_name}/pcd_saves/${PKL_FILENAME} \
            --class_names_file ${DATASET_ROOT}/${scene_label}/${scene_name}/gsa_classes_none.json


        python ${CG_FOLDER}/scenegraph/build_scenegraph_cfslam.py \
            --mode refine-node-captions \
            --cachedir ${DATASET_ROOT}/${scene_label}/${scene_name}/sg_cache \
            --mapfile ${DATASET_ROOT}/${scene_label}/${scene_name}/pcd_saves/${PKL_FILENAME} \
            --class_names_file ${DATASET_ROOT}/${scene_label}/${scene_name}/gsa_classes_none.json


        python ${CG_FOLDER}/scenegraph/build_scenegraph_cfslam.py \
            --mode build-scenegraph \
            --cachedir ${DATASET_ROOT}/${scene_label}/${scene_name}/sg_cache \
            --mapfile ${DATASET_ROOT}/${scene_label}/${scene_name}/pcd_saves/${PKL_FILENAME} \
            --class_names_file ${DATASET_ROOT}/${scene_label}/${scene_name}/gsa_classes_none.json

        python ${CG_FOLDER}/scenegraph/build_scenegraph_cfslam.py \
            --mode generate-scenegraph-json \
            --cachedir ${DATASET_ROOT}/${scene_label}/${scene_name}/sg_cache \
            --mapfile ${DATASET_ROOT}/${scene_label}/${scene_name}/pcd_saves/${PKL_FILENAME} \
            --class_names_file ${DATASET_ROOT}/${scene_label}/${scene_name}/gsa_classes_none.json

        # mkdir -p "${NEW_ROOT}/${scene_label}/${scene_name}/pcd_saves/"
        # cp -r "${DATASET_ROOT}/${scene_label}/${scene_name}/pcd_saves/" "${NEW_ROOT}/${scene_label}/${scene_name}/"
        # cp -r "${DATASET_ROOT}/${scene_label}/${scene_name}/gsa_classes_none.json" "${NEW_ROOT}/${scene_label}/${scene_name}/"

        # mkdir -p "${NEW_ROOT}/${scene_label}/${scene_name}/sg_cache/"
        # cp -r "${DATASET_ROOT}/${scene_label}/${scene_name}/sg_cache/scene_graph.json" "${NEW_ROOT}/${scene_label}/${scene_name}/sg_cache/"
    done
done