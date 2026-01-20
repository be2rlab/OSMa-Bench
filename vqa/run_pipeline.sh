#!/bin/bash

# Configuration and directories
CONFIG_PATH="./config/gemini_qa.yml"
DATA_DIR="./data"
GRAPHS_DIR="./graphs"
OUTPUT_DIR="./output"
EVAL_DIR="./evaluated"

# Ensure output directories exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$EVAL_DIR"

echo "=== STEP 1: GENERATION (descriptions → QA → validation) ==="
for SCENE_PATH in "$DATA_DIR"/*/; do
    SCENE_NAME=$(basename "$SCENE_PATH")
    echo "--- Processing scene: $SCENE_NAME"

    # Generate descriptions, questions and validate them
    python -m src.generation.text_desc_generation "$CONFIG_PATH" --scene "$SCENE_NAME" || continue
    python -m src.generation.qa_generation      "$CONFIG_PATH" --scene "$SCENE_NAME" || continue
    python -m src.validation.qa_validation "$CONFIG_PATH" --scene "$SCENE_NAME" || continue
done

echo "=== STEP 2: SCENE GRAPH ANSWERING ==="
for SCENE_PATH in "$DATA_DIR"/*/; do
    SCENE_NAME=$(basename "$SCENE_PATH")
    VQA_FILE="${SCENE_PATH}vqa/${SCENE_NAME}_validated_questions.json"
    GRAPH_FILE="${GRAPHS_DIR}/${SCENE_NAME}/scene_graph.json"
    OUTPUT_FILE="${OUTPUT_DIR}/${SCENE_NAME}_answered.json"

    if [[ ! -f "$VQA_FILE" ]] || [[ ! -f "$GRAPH_FILE" ]]; then
        echo "[!] Skipping: missing VQA or scene graph for $SCENE_NAME"
        continue
    fi

    python -m src.evaluation.scene_graph_answering --c "$CONFIG_PATH" \
                                        --questions "$VQA_FILE" \
                                        --graph     "$GRAPH_FILE" \
                                        --output    "$OUTPUT_FILE"
done

echo "=== STEP 3: EVALUATION ==="
python -m src.evaluation.graphs_evaluation "$CONFIG_PATH"

# Pause before exit
read -p "Press any key to exit..." -n1 -s
echo
