import argparse
import json
import os
import re
import logging
from tqdm import tqdm
from typing import List, Dict


from src.config import Configuration
from src.utils.api import request_gemini
from src.utils.json_utils import to_json_string, load_json, save_json, clean_json_response
from src.utils.parsing import build_scene_inventory

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

DIGIT_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10"
}
GENERIC_SPATIAL = re.compile(
    r"\b(on|at|in|on the|on a) (wall|floor|ceiling)\b", re.IGNORECASE
)
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
    # Build a dictionary { frame_name: description_text }
    return {entry["frame"]: entry["description"] for entry in data.get("parameters", [])}

def post_filter_qas(qa_list: list[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    1) Measurement: keep only those with numbers (convert words → numbers)
    2) Object Relations: Removing the overly general “on wall/floor/ceiling”
    """
    cleaned = []
    for qa in qa_list:
        cat = qa.get("category", "")
        ans = qa.get("answer", "").strip()
        # 1) Measurement → digit-only
        if cat == "Measurement":
            lower = ans.lower()
            if lower in DIGIT_MAP:
                qa["answer"] = DIGIT_MAP[lower]
            if not qa["answer"].isdigit():
                continue
        # 2) generic spatial
        if cat.startswith("Object Relations") and GENERIC_SPATIAL.search(qa.get("question", "")):
            continue
        cleaned.append(qa)
    return cleaned


def generate_questions(config, descriptions):
    question_prompt_str = config.qa_generation_prompt
    scene_inventory = build_scene_inventory(descriptions)
    inv_json = to_json_string(scene_inventory)

    qa_data = {}

    for frame, description in tqdm(descriptions.items(), desc="Generating questions"):
        if config.rejection_keyword in description:
            continue

        prompt = (
            f"{question_prompt_str}\n\n"
            "## Scene-level inventory (approx counts across all frames):\n"
            f"{inv_json}\n\n"
            "## Frame description:\n"
            f"{description}"
        )

        resp = request_gemini(config, prompt)
        if resp:
            cleaned = clean_json_response(resp)
            try:
                qa_list = json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning("Failed to parse QA JSON for frame %s; raw response:\n%s", frame, cleaned)
                qa_list = []

        qa_data[frame] = post_filter_qas(qa_list)
    return qa_data


def save_qa_data(output_path, scene_name, frames_order, qa_data):
    """
    Saves generated questions in a JSON structure under 'parameters'.
    Each element includes a 'frame' and a 'qa' list.
    """
    scene_data = {"scene_name": scene_name, "parameters": []}

    for frame in frames_order:
        if frame not in qa_data:
            continue
        scene_data["parameters"].append({
            "frame": frame,
            "qa": qa_data[frame]
        })

    save_json(scene_data, output_path)

    logger.info("QA data saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the YAML config.")
    parser.add_argument("--scene", required=True, help="Name of the scene folder.")
    args = parser.parse_args()

    config = Configuration(yaml_path=args.config_path)

    vqa_dir = os.path.join(config.base_scenes_dir, args.scene, "vqa")

    description_file = os.path.join(vqa_dir, f"{args.scene}_descriptions.json")

    if not os.path.isfile(description_file):
        raise FileNotFoundError(
            f"Scene descriptions file not found at: {description_file}. "
            "Please run the text_desc_generation script first."
        )

    descriptions = load_existing_descriptions(description_file)

    frames_order = list(descriptions.keys())

    qa_data = generate_questions(config, descriptions)

    # Save QA results
    output_json = os.path.join(vqa_dir, f"{args.scene}_questions.json")
    save_qa_data(output_json, args.scene, frames_order, qa_data)


if __name__ == "__main__":
    main()
