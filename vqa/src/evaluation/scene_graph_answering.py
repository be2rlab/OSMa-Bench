import argparse
import json
import logging

from src.config import Configuration
from src.utils.api import post_with_retry
from src.utils.json_utils import load_json, save_json, clean_json_response, extract_questions

logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )
logger = logging.getLogger(__name__)

def batch_scene_graph_answering(config, questions, scene_graph, batch_size=10):
    """
    Answer questions about the scene graph in batches via Gemini API.
    """
    answered = []
    api_url = f"{config.url}/{config.vlm}:generateContent?key={config.gemini_api_key}"

    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        prompt = (
            "Answer the following questions based ONLY on the provided scene graph.\n"
            "Add an 'answer' field for each question with your response.\n"
            "If Yes/No expected, answer strictly 'Yes' or 'No'.\n"
            "If counting, answer strictly as a number.\n\n"
            f"Scene graph: ```json\n{json.dumps(scene_graph)}\n```\n\n"
            f"Questions: ```json\n{json.dumps(batch)}\n```"
        )
        payload = {"contents":[{"parts":[{"text": prompt}]}]}

        resp = post_with_retry(api_url, payload)
        if not resp:
            logger.error("No response for batch %d/%d", i, len(questions))
            raise RuntimeError("Model did not return a response.")

        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        cleaned = clean_json_response(raw)

        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, list):
                raise ValueError("Parsed response is not a list")
            answered.extend(parsed)
            logger.debug("Batch %d parsed successfully: %d items", i, len(parsed))
        except Exception as e:
            logger.exception("Failed to parse JSON in batch %d: %s", i, e)
            raise

    return answered

def merge_answers(original_questions, answered_questions):
    """
    Merge model's answers back into the original question dicts.
    """
    merged = []
    for orig, ans in zip(original_questions, answered_questions):
        entry = orig.copy()
        entry["scene_graph_answer"] = ans.get("answer", "No answer")
        merged.append(entry)
    return merged

def main():
    parser = argparse.ArgumentParser("Answer VQA via scene graph")
    parser.add_argument("-c","--config", dest="cfg", required=True, help="YAML config path")
    parser.add_argument("--questions", required=True, help="Questions JSON")
    parser.add_argument("--graph", required=True, help="Scene graph JSON")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    config = Configuration(yaml_path=args.cfg)

    logger.info("Loading questions from %s", args.questions)
    qa_json = load_json(args.questions)
    questions = extract_questions(qa_json)
    if not questions:
        logger.error("No questions found in %s", args.questions)
        raise ValueError(f"No questions in {args.questions}")
    logger.info("Loaded %d questions", len(questions))

    logger.info("Loading scene graph from %s", args.graph)
    scene_graph = load_json(args.graph)

    logger.info("Answering questions via Gemini API")
    answered = batch_scene_graph_answering(
        config=config,
        questions=[{"question": q["question"]} for q in questions],
        scene_graph=scene_graph
    )

    result = merge_answers(questions, answered)
    logger.info("Saving merged answers to %s", args.output)
    save_json(result, args.output)

if __name__ == "__main__":
    main()
