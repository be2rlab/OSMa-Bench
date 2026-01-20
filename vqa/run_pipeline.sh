#!/bin/bash

CONFIG_PATH="./config/gemini_qa.yml"
DATA_DIR="./data/replica_cad"
GRAPHS_DIR="./graphs/baseline"


for SCENE_PATH in "$DATA_DIR"/*/; do
    SCENE_NAME=$(basename "$SCENE_PATH")
    GRAPH_FILE="${GRAPHS_DIR}/${SCENE_NAME}/sg_cache/scene_graph.json"

    echo "--- Processing scene: $SCENE_NAME"

    echo "=== STEP 1: GENERATION (descriptions → QA → validation) ==="
    python -m src.generation.text_desc_generation "$CONFIG_PATH" --scene "$SCENE_NAME"
    python -m src.generation.qa_generation      "$CONFIG_PATH" --scene "$SCENE_NAME"
    python -m src.validation.qa_validation "$CONFIG_PATH" --scene "$SCENE_NAME"

    echo "=== STEP 2: SCENE GRAPH ANSWERING ==="
    python -m src.evaluation.scene_graph_answering  "$CONFIG_PATH" \
                                        --scene     "$SCENE_NAME" \
                                        --graph     "$GRAPH_FILE"

    echo "=== STEP 3: EVALUATION ==="
    python -m src.evaluation.graphs_evaluation "$CONFIG_PATH" --scene "$SCENE_NAME"

    break
done