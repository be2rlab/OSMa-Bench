import argparse
import csv
import logging
import os
from collections import defaultdict
from typing import List, Dict, Tuple
from tqdm import trange

from src.models import create_foundation_model
from src.utils.api import post_with_retry
from src.utils.config_loader import load_configuration
from src.utils.json_utils import (
    load_json,
    save_json,
    to_json_string,
    parse_json
)
from src.utils.parsing import is_yes_no_answer, is_numeric_answer


logger = logging.getLogger(__name__)


def get_pathes(base_dir, scene_name):
    vqa_dir             = os.path.join(base_dir, scene_name, "vqa")
    answered_file_path  = os.path.join(vqa_dir, f"{scene_name}_answered.json")
    evaluated_file_path = os.path.join(vqa_dir, f"{scene_name}_evaluated.json")
    
    return answered_file_path, evaluated_file_path


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the YAML config.")
    parser.add_argument("--scene", required=True, help="Name of the scene folder.")
    
    args = parser.parse_args()
    
    return args


def evaluate_answers_locally(answered: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    local, to_llm = [], []
    
    for q in answered:
        gt_answer = str(q["answer"]).strip().lower()
        pred_answer = str(q.get("scene_graph_answer", "")).strip().lower()
        
        if is_yes_no_answer(gt_answer) and is_yes_no_answer(pred_answer):
            q["similar"] = "Yes" if gt_answer == pred_answer else "No"
            local.append(q)
        elif is_numeric_answer(gt_answer) and is_numeric_answer(pred_answer):
            q["similar"] = "Yes" if gt_answer == pred_answer else "No"
            local.append(q)
        else:
            to_llm.append(q)
            
    logger.info("Local eval: %d, to LLM: %d", len(local), len(to_llm))
    
    return local, to_llm


def evaluate_with_llm(
        foundation_model,
        prompt,
        questions: List[Dict],
        batch_size: int = 10
) -> List[Dict]:
    """Assessment of the remaining through LLM."""
    results = []
    
    for i in trange(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        batch_prompt = prompt.replace('{questions}', to_json_string(batch))
        
        resp = post_with_retry(foundation_model, batch_prompt)
        
        if not resp:
            logger.error("No response for batch %d/%d", i, len(questions))
            continue

        parsed = parse_json(resp)

        if parsed is None:
            logger.error("Failed to parse JSON in batch %d", i)
            continue

        if isinstance(parsed, list):
            results.extend(parsed)
            logger.debug("LLM batch %d parsed %d entries", i, len(parsed))
        else:
            logger.error("Failed to parse LLM output for batch %d", i)
            
    return results


def process_file(answered_file_path, evaluated_file_path, foundation_model, prompt, batch_size=10) -> None:
    answered = load_json(answered_file_path)
    
    locally_eval, to_llm = evaluate_answers_locally(answered)
    llm_eval = evaluate_with_llm(foundation_model, prompt, to_llm, batch_size=batch_size)
    
    combined = locally_eval + llm_eval

    save_json(combined, evaluated_file_path)


def main():
    args = get_parser_args()

    config = load_configuration(yaml_path=args.config_path)
    
    answered_file_path, evaluated_file_path = get_pathes(
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
    
    answered = load_json(answered_file_path)
    
    locally_eval, to_llm = evaluate_answers_locally(answered)
    
    eval_conf = config.evaluation
    llm_eval = evaluate_with_llm(foundation_model, eval_conf.prompt, to_llm, batch_size=eval_conf.batch_size)
    
    combined = locally_eval + llm_eval
    save_json(combined, evaluated_file_path)


if __name__ == "__main__":
    main()
