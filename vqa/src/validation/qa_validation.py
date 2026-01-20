import os
import argparse

from src.config import Configuration
from src.utils.json_utils import save_json
from src.validation.validation_utils import (
    load_scene_qa,
    get_scene_description,
    build_scene_counts,
    filter_frequent_objects,
    filter_duplicates_and_conflicts,
    iterative_neural_validation,
    remove_wrong_measurement_questions,
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Validate QA for one scene")
    parser.add_argument("config_path", help="Path to YAML config")
    parser.add_argument("--scene", required=True, help="Scene folder name")
    args = parser.parse_args()

    config = Configuration(yaml_path=args.config_path)
    base = config.base_scenes_dir
    scene = args.scene
    results_dir = os.path.join(base, scene, "results")
    vqa_dir     = os.path.join(base, scene, "vqa")
    os.makedirs(vqa_dir, exist_ok=True)

    scene_qa = load_scene_qa(vqa_dir, scene)
    scene_desc = get_scene_description(os.path.join(vqa_dir, f"{scene}_descriptions.json"))

    scene_counts = build_scene_counts(scene_desc)
    with open(os.path.join(vqa_dir, "object_counts_from_description.log"), "w") as f:
        for obj, cnt in scene_counts.items():
            f.write(f"{obj}: {cnt}\n")

    qa_step1 = filter_frequent_objects(config, scene_qa, vqa_dir)

    qa_step2 = filter_duplicates_and_conflicts(qa_step1, vqa_dir)

    qa_step3 = iterative_neural_validation(
        config=config,
        qa_by_frame=qa_step2,
        results_dir=results_dir,
        scene_description=scene_desc,
        vqa_dir=vqa_dir,
        max_iterations=5
    )

    qa_step4 = remove_wrong_measurement_questions(qa_step3, scene_counts, vqa_dir)

    output_file = os.path.join(vqa_dir, f"{scene}_validated_questions.json")
    save_json(
        {"scene_name": scene, "parameters": [
            {"frame": frame, "qa": qa_list}
            for frame, qa_list in qa_step4.items()
        ]},
        output_file
    )
    logger.info("Validation complete, results saved to %s", output_file)

if __name__ == "__main__":
    main()
