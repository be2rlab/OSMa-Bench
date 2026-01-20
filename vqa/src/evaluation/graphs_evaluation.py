import csv
import logging
import os
import re
from collections import defaultdict
from typing import List, Dict, Tuple

from src.config import Configuration
from src.utils.api import post_with_retry
from src.utils.json_utils import (
    load_json,
    save_json,
    to_json_string,
    clean_json_response,
    parse_json
)

OUTPUT_DIR = "./output"
EVALUATED_DIR = "./evaluated"
METRICS_FILE = os.path.join(EVALUATED_DIR, "metrics.csv")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


def is_yes_no_answer(answer: str) -> bool:
    return answer.lower() in {"yes", "no"}


def is_numeric_answer(answer: str) -> bool:
    return bool(re.match(r"^\d+(\.\d+)?$", str(answer)))


def prepare_metrics_csv(all_categories: List[str]) -> None:
    """Creates a CSV"""
    os.makedirs(EVALUATED_DIR, exist_ok=True)
    with open(METRICS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["File Name", "Overall Accuracy", "Accuracy (No Spatial)"] + sorted(all_categories)
        )
    logger.info("Metrics CSV initialized with categories: %s", all_categories)


def evaluate_answers_locally(
        answered: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    local, to_llm = [], []
    for q in answered:
        gt = str(q["answer"]).strip().lower()
        pred = str(q.get("scene_graph_answer", "")).strip().lower()
        if is_yes_no_answer(gt) and is_yes_no_answer(pred):
            q["similar"] = "Yes" if gt == pred else "No"
            local.append(q)
        elif is_numeric_answer(gt) and is_numeric_answer(pred):
            q["similar"] = "Yes" if gt == pred else "No"
            local.append(q)
        else:
            to_llm.append(q)
    logger.info("Local eval: %d, to LLM: %d", len(local), len(to_llm))
    return local, to_llm


def evaluate_with_llm(
        config: Configuration,
        questions: List[Dict],
        batch_size: int = 10
) -> List[Dict]:
    """Assessment of the remaining through LLM."""
    results = []
    api_url = f"{config.url}/{config.vlm}:generateContent?key={config.gemini_api_key}"
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        prompt = (
            "Compare ground truth answers ('answer') and scene-graph answers ('scene_graph_answer'). "
            "Add a 'similar':'Yes'/'No' field to each entry.\n"
            f"Questions: ```json\n{to_json_string(batch)}\n```"
        )
        resp = post_with_retry(api_url, {"contents": [{"parts": [{"text": prompt}]}]})
        if not resp:
            logger.warning("No LLM response for batch %d", i)
            continue

        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        cleaned = clean_json_response(raw)
        parsed = parse_json(cleaned)
        if isinstance(parsed, list):
            results.extend(parsed)
            logger.debug("LLM batch %d parsed %d entries", i, len(parsed))
        else:
            logger.error("Failed to parse LLM output for batch %d", i)
    return results


def compute_metrics(
        evaluated: List[Dict]
) -> Tuple[float, Dict[str, float], float]:
    total_sim, total_dis = 0, 0
    cat_counts = defaultdict(lambda: {"sim": 0, "dis": 0})
    sim_ns, dis_ns = 0, 0

    for q in evaluated:
        sim = (q["similar"].lower() == "yes")
        cat = q.get("category", "Unknown")
        if sim:
            total_sim += 1
            cat_counts[cat]["sim"] += 1
            if cat != "Object Relations - Spatial":
                sim_ns += 1
        else:
            total_dis += 1
            cat_counts[cat]["dis"] += 1
            if cat != "Object Relations - Spatial":
                dis_ns += 1

    overall = total_sim / (total_sim + total_dis) if total_sim + total_dis > 0 else 0
    no_spatial = sim_ns / (sim_ns + dis_ns) if sim_ns + dis_ns > 0 else 0
    per_cat = {
        cat: v["sim"] / (v["sim"] + v["dis"])
        for cat, v in cat_counts.items() if v["sim"] + v["dis"] > 0
    }
    return overall, per_cat, no_spatial


def save_metrics_row(
        file_name: str,
        overall: float,
        per_cat: Dict[str, float],
        no_spatial: float,
        all_categories: List[str]
) -> None:
    with open(METRICS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        row = [file_name, f"{overall:.2%}", f"{no_spatial:.2%}"]
        row += [f"{per_cat.get(cat, 0):.2%}" for cat in sorted(all_categories)]
        writer.writerow(row)
    logger.info("Metrics for %s saved.", file_name)


def process_file(config: Configuration, file: str, all_categories: List[str]) -> None:
    inp = os.path.join(OUTPUT_DIR, file)
    out = os.path.join(EVALUATED_DIR, file)

    answered = load_json(inp)
    local, to_llm = evaluate_answers_locally(answered)
    llm_res = evaluate_with_llm(config, to_llm)
    combined = local + llm_res

    save_json(combined, out)
    overall, per_cat, no_sp = compute_metrics(combined)
    save_metrics_row(file, overall, per_cat, no_sp, all_categories)


def main():
    import argparse
    parser = argparse.ArgumentParser("Evaluate scene-graph VQA")
    parser.add_argument("config_path", help="YAML config")
    args = parser.parse_args()

    config = Configuration(yaml_path=args.config_path)

    cats = set()
    for fn in os.listdir(OUTPUT_DIR):
        if fn.endswith(".json"):
            for q in load_json(os.path.join(OUTPUT_DIR, fn)):
                cats.add(q.get("category", "Unknown"))

    prepare_metrics_csv(list(cats))

    for fn in sorted(os.listdir(OUTPUT_DIR)):
        if fn.endswith(".json"):
            logger.info("Processing %s", fn)
            process_file(config, fn, list(cats))


if __name__ == "__main__":
    main()
