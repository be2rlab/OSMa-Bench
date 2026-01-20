import argparse
import logging
import os
import re
from collections import Counter
from typing import List, Dict

from tqdm import tqdm

from src.models import create_foundation_model
from src.utils.api import post_with_retry
from src.utils.config_loader import load_configuration
from src.utils.json_utils import load_json, parse_json, to_json_string, save_json


logger = logging.getLogger(__name__)


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the YAML config.")
    parser.add_argument("--scene", required=True, help="Name of the scene folder.")
    
    args = parser.parse_args()
    
    return args


def get_pathes(base_dir, scene_name):
    vqa_dir         = os.path.join(base_dir, scene_name, "vqa")
    desc_file_path  = os.path.join(vqa_dir, f"{scene_name}_descriptions.json")
    
    return vqa_dir, desc_file_path


def load_existing_descriptions(description_file):
    """
    Loads scene descriptions from a JSON file.
    Raises an error if the file does not exist.
    """
    if not os.path.exists(description_file):
        raise FileNotFoundError(
            f"Error: {description_file} not found. Please run text_desc_generation.py first."
        )
        
    data = load_json(description_file)

    return {entry["frame"]: entry["description"] for entry in data.get("parameters", [])}


def get_digit_map():
    return {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10"
    }
    

def post_filter_qas(qa_list: list[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    1) Measurement: keep only those with numbers (convert words → numbers)
    2) Object Relations: Removing the overly general “on wall/floor/ceiling”
    """
    
    digit_map = get_digit_map()
    
    spatial_re = re.compile(
        r"\b(on|at|in|on the|on a) (wall|floor|ceiling)\b", re.IGNORECASE
    )
    
    filtered = []
    for qa in qa_list:
        category = str(qa.get("category", '')).strip()
        question = str(qa.get("question", '')).strip()
        answer = str(qa.get("answer", '')).strip()
        
        if not category or not question or not answer:
            continue

        if category == "Measurement":
            qa["answer"] = digit_map.get(answer.lower(), answer.lower())

            if not qa["answer"].isdigit():
                continue
            
        elif category.startswith("Object Relations") and spatial_re.search(question):
            continue
        
        filtered.append(qa)
        
    return filtered


def build_scene_inventory(desc_dict: dict) -> Counter:
    """
    Count approximate number of objects across all frames.
    desc_dict: mapping frame_name -> description_text.
    Returns a Counter { object_name: total_count }.
    """
    inventory = Counter()
    
    digit_map = get_digit_map()
    
    # pattern matches digits or words from DIGIT_MAP followed by an object name
    pattern = re.compile(
        rf"\b(\d+|{'|'.join(digit_map.keys())})\s+([\w\-]+)\b", re.I
    )
    
    for description in desc_dict.values():
        for num, obj in pattern.findall(description):
            if num.isdigit():
                n = int(num)
            else:
                n = int(digit_map[num.lower()])
            inventory[obj.lower()] += n
            
    return inventory


def generate_questions(foundation_model, descriptions, model_prompt, rejection_keyword):
    scene_inventory = build_scene_inventory(descriptions)
    inv_json = to_json_string(scene_inventory)

    qa_data = {}

    for frame, description in tqdm(descriptions.items(), desc="Generating questions"):
        if rejection_keyword in description:
            continue

        prompt = (
            f"{model_prompt}\n\n"
            "## Scene-level inventory (approx counts across all frames):\n"
            f"{inv_json}\n\n"
            "## Frame description:\n"
            f"{description}"
        )
        
        resp = post_with_retry(foundation_model, prompt=prompt)
        
        if resp is None:
            qa_data[frame] = []
            logger.warning("Failed to get model response for frame %s", frame)
            continue

        qa_list = parse_json(resp)
        
        if qa_list is None:
            qa_data[frame] = []
            logger.warning("Failed to parse QA JSON for frame %s", frame)
            continue

        qa_data[frame] = post_filter_qas(qa_list)
        
    return qa_data


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

    vqa_dir, desc_file_path = get_pathes(
        base_dir=config.data_dir, scene_name=args.scene
    )
    
    descriptions = load_existing_descriptions(description_file=desc_file_path)
    
    model_conf = config.foundation_model
    foundation_model = create_foundation_model(
        api_name = model_conf.api_name,
        api_key = model_conf.api_key,
        model = model_conf.model,
        llm = model_conf.llm,
        lvlm = model_conf.lvlm,
        )

    qa_data = generate_questions(
        foundation_model = foundation_model, 
        descriptions = descriptions,
        model_prompt = config.qa_generation.prompt,
        rejection_keyword = config.description_generation.rejection_keyword
    )

    save_qa_data(
        output_path = os.path.join(vqa_dir, f"{args.scene}_questions.json"), 
        scene_name = args.scene,
        qa_data = qa_data
    )


if __name__ == "__main__":
    main()
