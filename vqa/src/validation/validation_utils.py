import os
import re
import time
import json
import numpy as np
from collections import Counter
from tqdm import tqdm

from src.utils.api import post_with_retry
from src.generation.text_desc_generation import encode_image
from src.utils.parsing import infer_answer_type
from src.utils.json_utils import load_json

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def load_scene_qa(vqa_dir: str, scene_name: str) -> dict:
    """
    Load QA pairs from vqa/<scene_name>_questions.json.
    Returns {frame: [qa_dict, ...], ...}.
    """
    path = os.path.join(vqa_dir, f"{scene_name}_questions.json")
    data = load_json(path)
    return {p["frame"]: p.get("qa", []) for p in data.get("parameters", [])}


def get_scene_description(description_file: str) -> str:
    """
    Concatenate all 'description' fields from descriptions JSON.
    """
    data = load_json(description_file)
    return " ".join(p["description"] for p in data.get("parameters", []))


def build_scene_counts(scene_description: str) -> Counter:
    """
    Count occurrences like '3 chairs' in the combined scene description.
    """
    pattern = re.compile(r"\b(\d+)\s+([\w\-]+)")
    counts = Counter()
    for num, obj in pattern.findall(scene_description):
        counts[obj.lower()] += int(num)
    return counts


def validate_qa(config, image_path: str, qa_list: list, scene_description: str,
                log_file: str, batch_size: int = 5) -> list:
    """
    Send QA in batches to Gemini VLM for validation.
    Returns a list of validated QA dicts.
    """
    time.sleep(2)
    api_url = f"{config.url}/{config.vlm}:generateContent?key={config.gemini_api_key}"
    prompt = config.validation_prompt
    validated = []

    for i in range(0, len(qa_list), batch_size):
        batch = qa_list[i : i + batch_size]
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType":"image/jpeg","data":encode_image(image_path)}},
                    {"text": "Scene Description:\n" + scene_description},
                    {"text": json.dumps(batch)}
                ]
            }]
        }
        resp = post_with_retry(api_url, payload)
        if not resp:
            logger.warning("No response for %s batch %d", image_path, i)
            continue

        # Extract and clean the returned JSON
        cand = resp.json().get("candidates", [{}])[0]
        parts = cand.get("content", {}).get("parts", [{}])
        raw = parts[0].get("text", "") if parts else ""
        cleaned = re.sub(r"```json\s*(.*?)\s*```", r"\1", raw, flags=re.DOTALL).strip()

        # Log raw VLM response
        with open(log_file, "a", encoding="utf-8") as lg:
            lg.write(f"\n=== VLM response for {os.path.basename(image_path)} batch {i} ===\n")
            lg.write(cleaned + "\n")

        try:
            parsed = json.loads(cleaned)
            validated.extend(parsed)
        except json.JSONDecodeError as e:
            logger.warning("JSON parse error in batch %d: %s", i, e)

    return validated


def filter_non_objects(config, word_list: list) -> set:
    """
    Use LLM to filter out non-object words from a list.
    Returns a set of object words.
    """
    time.sleep(2)
    api_url = f"{config.url}/{config.vlm}:generateContent?key={config.gemini_api_key}"
    prompt = config.filter_non_objects_prompt
    payload = {"contents":[{"parts":[{"text": prompt}, {"text": json.dumps(word_list)}]}]}

    resp = post_with_retry(api_url, payload)
    if not resp:
        return set()

    cand = resp.json().get("candidates", [{}])[0]
    parts = cand.get("content", {}).get("parts", [{}])
    raw = parts[0].get("text", "") if parts else ""
    cleaned = re.sub(r"```json\s*(.*?)\s*```", r"\1", raw, flags=re.DOTALL).strip()

    try:
        result = json.loads(cleaned)
        return set(result) if isinstance(result, list) else set()
    except json.JSONDecodeError:
        return set()


def filter_frequent_objects(config, validated_qa, vqa_dir):
    """
    Remove questions about objects that appear too frequently.
    Compute 3×median threshold and drop excess.
    """
    # Collect all words from questions
    all_objects = []
    for qa_list in validated_qa.values():
        for qa in qa_list:
            all_objects.extend(re.findall(r"\b\w+\b", qa["question"].lower()))

    # Identify actual object words via LLM
    unique_words = list(set(all_objects))
    filtered_objects = filter_non_objects(config, unique_words)

    # Count how often each object appears
    object_counts = Counter(w for w in all_objects if w in filtered_objects)

    # Log counts before filtering
    with open(os.path.join(vqa_dir, "filtered_objects_before.txt"), "w") as log:
        log.write("Filtered Objects BEFORE filtering:\n")
        for obj, cnt in object_counts.most_common():
            log.write(f"{obj}: {cnt}\n")

    # Determine adaptive threshold (3×median)
    if object_counts:
        median_count = np.median(list(object_counts.values()))
        threshold = median_count * 3
    else:
        median_count = 0
        threshold = 0

    overused_objects = {obj: cnt for obj, cnt in object_counts.items() if cnt > threshold}

    # Perform filtering: for each frame, drop questions if an object is overused
    removed_questions = []
    filtered_qa = {}
    for frame, qa_list in validated_qa.items():
        new_list = []
        for qa in qa_list:
            q_text = qa["question"].lower()
            # find overused objects in this question
            hits = [o for o in overused_objects if o in q_text]
            if hits:
                related = [x for x in qa_list if any(o in x["question"].lower() for o in hits)]
                # if too many related questions, drop this one
                if len(related) > 5:
                    removed_questions.append((frame, qa["question"], "overused_object"))
                    continue
            new_list.append(qa)
        filtered_qa[frame] = new_list

    # Log counts after filtering
    filtered_objects_list = []
    for qa_list in filtered_qa.values():
        for qa in qa_list:
            filtered_objects_list.extend(re.findall(r"\b\w+\b", qa["question"].lower()))
    after_counts = Counter(w for w in filtered_objects_list if w in filtered_objects)

    with open(os.path.join(vqa_dir, "filtered_objects_after.txt"), "w") as log:
        log.write("Filtered Objects AFTER filtering:\n")
        for obj, cnt in after_counts.most_common():
            log.write(f"{obj}: {cnt}\n")

    # Append removal log
    with open(os.path.join(vqa_dir, "removed_questions.log"), "a") as log:
        for frame, question, reason in removed_questions:
            log.write(f"Frame: {frame}\nQuestion: {question}\nReason: {reason}\n\n")

    logger.info("Frequency filter removed %d questions", len(removed_questions))

    return filtered_qa



def filter_duplicates_and_conflicts(qa_by_frame: dict, vqa_dir: str) -> dict:
    """
    Remove exact duplicates and resolve conflicts:
      - Prefer 'Yes' over 'No' for boolean
      - For numeric, keep the highest
    """
    filtered = {f: [] for f in qa_by_frame}
    variants = {}
    removed = []

    for frame, qa_list in qa_by_frame.items():
        for qa in qa_list:
            q_text = qa["question"]
            ans    = qa["answer"]
            typ    = infer_answer_type(ans)

            if q_text not in variants:
                variants[q_text] = [{"answer":ans,"frame":frame,"type":typ}]
                filtered[frame].append(qa)
                continue

            existing = variants[q_text]
            if ans in [v["answer"] for v in existing]:
                removed.append((frame, q_text, "duplicate"))
                continue

            # Boolean: prefer Yes
            if typ == "boolean":
                if any(v["type"]=="boolean" and v["answer"].lower()=="yes" for v in existing):
                    if ans.lower()=="no":
                        removed.append((frame, q_text, "prefer Yes"))
                        continue
                if ans.lower()=="yes":
                    # drop prior No
                    for v in existing:
                        if v["type"]=="boolean" and v["answer"].lower()=="no":
                            filtered[v["frame"]] = [
                                x for x in filtered[v["frame"]]
                                if x["question"] != q_text
                            ]
                            removed.append((v["frame"], q_text, "replaced No"))
                            variants[q_text].remove(v)

            # Numeric: keep max
            if typ == "numeric":
                try:
                    val_new = float(ans)
                except:
                    val_new = None
                vals_old = [float(v["answer"]) for v in existing if v["type"]=="numeric"]
                if vals_old and val_new is not None:
                    if val_new <= max(vals_old):
                        removed.append((frame, q_text, "lower numeric"))
                        continue
                    else:
                        # remove older lower
                        for v in existing:
                            if v["type"]=="numeric":
                                filtered[v["frame"]] = [
                                    x for x in filtered[v["frame"]]
                                    if not (x["question"]==q_text and x["answer"]==v["answer"])
                                ]
                                removed.append((v["frame"], q_text, "replaced numeric"))
                        variants[q_text] = [v for v in variants[q_text] if v["type"]!="numeric"]

            variants[q_text].append({"answer":ans,"frame":frame,"type":typ})
            filtered[frame].append(qa)

    # Log removals
    with open(os.path.join(vqa_dir, "removed_questions.log"), "a") as lg:
        for frame, q, reason in removed:
            lg.write(f"Frame: {frame}\nQuestion: {q}\nReason: {reason}\n\n")

    logger.info("Removed %d duplicated/conflict questions", len(removed))

    return filtered


def remove_wrong_measurement_questions(qa_by_frame: dict,
                                       scene_counts: Counter,
                                       vqa_dir: str) -> dict:
    """
    Drop Measurement questions whose numeric answer mismatches the scene_counts.
    """
    pattern = re.compile(r"How many ([\w\s\-]+?) (?:are|is) ", re.I)
    cleaned = {}

    for frame, qa_list in qa_by_frame.items():
        new_list = []
        for qa in qa_list:
            if qa.get("category") == "Measurement":
                try:
                    num = int(qa["answer"].replace(",", ""))
                except:
                    new_list.append(qa)
                    continue
                m = pattern.match(qa["question"])
                if m:
                    obj = m.group(1).strip().lower()
                    exp = scene_counts.get(obj)
                    if exp is not None and exp != num:
                        with open(os.path.join(vqa_dir, "removed_questions.log"), "a") as lg:
                            lg.write(
                                f"Frame: {frame}\n"
                                f"Question: {qa['question']}\n"
                                f"Answer: {qa['answer']}  Expected: {exp}\n"
                                "Reason: wrong count\n\n"
                            )
                        continue
            new_list.append(qa)
        cleaned[frame] = new_list

    return cleaned


def iterative_neural_validation(
    config,
    qa_by_frame: dict,
    results_dir: str,
    scene_description: str,
    vqa_dir: str,
    max_iterations: int = 5
) -> dict:
    """
    Iteratively call validate_qa() until no changes or max_iterations reached.
    """
    log_file = os.path.join(vqa_dir, "validation_process.log")
    current = qa_by_frame

    for i in range(max_iterations):
        logger.info("Neural validation iteration %d/%d", i+1, max_iterations)
        updated = {}
        total_before = sum(len(v) for v in current.values())
        any_changed = False

        for frame, qa_list in tqdm(current.items(), desc="Validating QA"):
            img_path = os.path.join(results_dir, frame)
            if not os.path.isfile(img_path):
                updated[frame] = qa_list
                continue

            raw_results = validate_qa(
                config, img_path, qa_list, scene_description, log_file
            )

            validated = [
                q for q in raw_results
                if isinstance(q, dict) and "question" in q and "answer" in q
            ]

            original = [
                q for q in qa_list
                if isinstance(q, dict) and "question" in q and "answer" in q
            ]

            set_validated = {(q["question"], q["answer"]) for q in validated}
            set_original  = {(q["question"], q["answer"]) for q in original}
            if set_validated != set_original:
                any_changed = True

            # If nothing valid came back, keep the previous list
            updated[frame] = validated or qa_list

        total_after = sum(len(v) for v in updated.values())
        removed = total_before - total_after
        logger.info("Iteration %d: removed %d questions", i+1, removed)
        if not any_changed:
            break

        current = updated

    return current

