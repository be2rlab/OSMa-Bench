import argparse
import logging
import os
from copy import deepcopy
from tqdm import trange

from src.models import create_foundation_model
from src.utils.config_loader import load_configuration
from src.utils.api import post_with_retry
from src.utils.json_utils import load_json, save_json, to_json_string, parse_json


logger = logging.getLogger(__name__)


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the YAML config.")
    parser.add_argument("--scene", required=True, help="Name of the scene folder.")
    parser.add_argument("--graph", required=True, help="Scene graph JSON")
    
    args = parser.parse_args()
    
    return args


def get_pathes(base_dir, scene_name):
    vqa_dir             = os.path.join(base_dir, scene_name, "vqa")
    val_qa_file_path    = os.path.join(vqa_dir, f"{scene_name}_validated_questions.json")
    answered_file_path  = os.path.join(vqa_dir, f"{scene_name}_answered.json")
    
    return val_qa_file_path, answered_file_path


def extract_questions(qa_json):
    """
    Flatten the nested QA structure under {"parameters": [{"frame":..., "qa":[...]}, ...]}
    into a list of {'question':..., 'answer':..., ...}.
    """
    if not isinstance(qa_json, dict) or "parameters" not in qa_json:
        raise ValueError("Invalid QA JSON format: missing 'parameters' key.")
    
    questions = []
    for frame_block in qa_json["parameters"]:
        for item in frame_block.get("qa", []):
            questions.append(item)

    questions = [dict(qa, **{'id': i}) for i, qa in enumerate(questions)]
    
    return questions


def load_questions(val_qa_file_path):
    logger.info("Loading questions from %s", val_qa_file_path)
    qa_json = load_json(val_qa_file_path)
    questions = extract_questions(qa_json)
    
    if not questions:
        logger.error("No questions found in %s", val_qa_file_path)
        raise ValueError(f"No questions in {val_qa_file_path}")
    
    logger.info("Loaded %d questions", len(questions))

    return questions


def remove_broken_answers(q_batch, ans_q_batch):
    filtered = []
    
    for answered in ans_q_batch:
        if not all(key in answered for key in ['id', 'question', 'answer']):
            logger.error(f"Parsed JSON miss key for question: {answered}")
            continue
        
        ident = answered.get('id', None)
        question = answered.get('question', None)
        
        q_with_id = {'id': ident, 'question': question}
        
        if q_with_id in q_batch:
            q_with_id['answer'] = answered.get('answer', None)
            
            filtered.append(q_with_id)
    
    return filtered


def batch_scene_graph_answering(foundation_model, prompt, questions, scene_graph, batch_size=10):
    """
    Answer questions about the scene graph in batches.
    """
    logger.info("Answering questions using foundation model")

    prompt = prompt.replace('{scene_graph}', to_json_string(scene_graph))

    answered = []
    
    for i in trange(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        batch_prompt = prompt.replace('{questions}', to_json_string(batch)) 

        resp = post_with_retry(foundation_model, batch_prompt)
        
        if not resp:
            logger.error("No response for batch %d/%d", i, len(questions))
            continue

        parsed = parse_json(resp)

        if parsed is None:
            logger.error("Failed to parse JSON in batch %d", i)
            continue

        if isinstance(parsed, dict):
            parsed = [parsed]
            
        filtered = remove_broken_answers(batch, parsed)

        answered.extend(filtered)

    return answered


def merge_answers(gt_questions, answered_questions):
    """
    Merge model's answers back into the original question dicts.
    """
    merged = deepcopy(gt_questions)

    ans_by_id = {entry['id']: entry['answer'] for entry in answered_questions}

    for entry in merged:
        entry["scene_graph_answer"] = ans_by_id.get(entry['id'], "**NO ANSWER**")
        entry.pop('id')

    return merged


def main():
    args = get_parser_args()

    config = load_configuration(yaml_path=args.config_path)
    
    val_qa_file_path, answered_file_path = get_pathes(
        base_dir=config.data_dir, scene_name=args.scene
    )
    
    questions = load_questions(val_qa_file_path)

    logger.info("Loading scene graph from %s", args.graph)
    scene_graph = load_json(args.graph)

    model_conf = config.foundation_model
    foundation_model = create_foundation_model(
        api_name = model_conf.api_name,
        api_key = model_conf.api_key,
        model = model_conf.model,
        llm = model_conf.llm,
        lvlm = model_conf.lvlm,
    )

    answering_conf = config.cg_answering
    answered = batch_scene_graph_answering(
        foundation_model = foundation_model,
        prompt = answering_conf.prompt,
        questions = [{key: q[key] for key in ['id', 'question']} for q in questions],
        scene_graph = scene_graph,
        batch_size = answering_conf.batch_size
    )

    result = merge_answers(questions, answered)
    
    logger.info("Saving merged answers to %s", answered_file_path)
    save_json(result, answered_file_path)


if __name__ == "__main__":
    main()
