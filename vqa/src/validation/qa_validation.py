import argparse
import logging
import os

from src.models import create_foundation_model
from src.utils.config_loader import load_configuration
from src.utils.json_utils import load_json, save_json
from src.validation.validation_utils import (
    build_scene_counts,
    filter_frequent_objects,
    filter_duplicates_and_conflicts,
    iterative_neural_validation,
    remove_wrong_measurement_questions,
)


logger = logging.getLogger(__name__)


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the YAML config.")
    parser.add_argument("--scene", required=True, help="Name of the scene folder.")
    
    args = parser.parse_args()
    
    return args


def get_pathes(base_dir, scene_name):
    results_dir     = os.path.join(base_dir, scene_name, "results")
    vqa_dir         = os.path.join(base_dir, scene_name, "vqa")
    desc_file_path  = os.path.join(vqa_dir, f"{scene_name}_descriptions.json")
    qa_file_path    = os.path.join(vqa_dir, f"{scene_name}_questions.json")
    
    return results_dir, vqa_dir, desc_file_path, qa_file_path


def load_scene_qa(qa_file) -> dict:
    """
    Load QA pairs from vqa/<scene_name>_questions.json.
    Returns {frame: [qa_dict, ...], ...}.
    """
    if not os.path.exists(qa_file):
        raise FileNotFoundError(
            f"Error: {qa_file} not found. Please run qa_generation.py first."
        )
    
    data = load_json(qa_file)
    
    return {entry["frame"]: entry.get("qa", []) for entry in data.get("parameters", [])}


def get_scene_description(description_file: str) -> str:
    """
    Concatenate all 'description' fields from descriptions JSON.
    """
    
    if not os.path.exists(description_file):
        raise FileNotFoundError(
            f"Error: {description_file} not found. Please run text_desc_generation.py first."
        )
    
    data = load_json(description_file)
    
    return " ".join(p["description"] for p in data.get("parameters", []))


def save_qa_data(output_path, scene_name, qa_data):
    """
    Saves generated questions in a JSON structure under 'parameters'.
    Each element includes a 'frame' and a 'qa' list.
    """
    scene_data = {"scene_name": scene_name, "parameters": []}

    for frame, qa in qa_data.items():        
        scene_data["parameters"].append({
            "frame": frame,
            "qa": qa
        })

    save_json(scene_data, output_path)

    logger.info("QA data saved to %s", output_path)


def main():
    args = get_parser_args()

    config = load_configuration(yaml_path=args.config_path)
    
    results_dir, vqa_dir, desc_file_path, qa_file_path = get_pathes(
        base_dir=config.data_dir, scene_name=args.scene
    )
    
    model_conf = config.foundation_model
    foundation_model = create_foundation_model(
        api_name = model_conf.api_name,
        api_key = model_conf.api_key,
        model = model_conf.model,
        llm = model_conf.llm,
        lvlm = model_conf.lvlm,
        )

    scene_desc = get_scene_description(desc_file_path)
    scene_qa = load_scene_qa(qa_file_path)

    val_conf = config.qa_validation
    filtered_qa = filter_frequent_objects(
        foundation_model = foundation_model, 
        prompt = val_conf.filter_non_objects_prompt, 
        qa_by_frame = scene_qa
    )

    filtered_qa = filter_duplicates_and_conflicts(qa_by_frame=filtered_qa)

    filtered_qa = iterative_neural_validation(
        foundation_model = foundation_model,
        prompt = val_conf.neural_val_prompt,
        qa_by_frame = filtered_qa,
        scene_description = scene_desc,
        max_iterations = 5
    )

    scene_counts = build_scene_counts(scene_desc)
    filtered_qa = remove_wrong_measurement_questions(filtered_qa, scene_counts, vqa_dir)
    
    output_path = os.path.join(vqa_dir, f"{args.scene}_validated_questions.json")
    save_qa_data(
        output_path = output_path,
        scene_name = args.scene,
        qa_data = filtered_qa
    )
    
    logger.info("Validation complete, results saved to %s", output_path)


if __name__ == "__main__":
    main()
