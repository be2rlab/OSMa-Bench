import json
import logging
import os
import random
import re
from collections import Counter
from copy import deepcopy

import numpy as np
from tqdm import tqdm, trange

from src.utils.api import post_with_retry
from src.utils.json_utils import to_json_string, parse_json
from src.utils.parsing import infer_answer_type


logger = logging.getLogger(__name__)


def build_scene_counts(scene_description: str) -> Counter:
    """
    Count occurrences like '3 chairs' in the combined scene description.
    """
    pattern = re.compile(r"\b(\d+)\s+([\w\-]+)")
    counts = Counter()
    for num, obj in pattern.findall(scene_description):
        counts[obj.lower()] += int(num)
    return counts


def validate_qa(foundation_model, prompt, image_path: str, qa_list: list, scene_description: str,
                batch_size: int = 5) -> list:
    """
    Send QA in batches to Gemini VLM for validation.
    Returns a list of validated QA dicts.
    """
    validated = []

    for i in trange(0, len(qa_list), batch_size):
        batch = qa_list[i : i + batch_size]
        
        # print(batch)
        
        full_prompt = (
            f"{prompt}\n\n"
            "Scene description:\n"
            f"{scene_description}\n\n"
            "Data:\n"
            f"{json.dumps(batch)}\n\n"
            "WRITE A JSON FORMAT WITH LIST [] ON TOP LEVEL"
        )

        resp = post_with_retry(foundation_model, prompt=full_prompt, images=[image_path])
        
        # print(resp)
        
        if resp is None:
            logger.warning("No response for %s batch %d", image_path, i // batch_size)
            continue
        
        qa_batch = parse_json(resp)
        
        if qa_batch is None:
            logger.warning("Failed to parse JSON for frame %s and batch %d", image_path, i // batch_size)
            continue

        validated.extend(qa_batch)

    return validated


def filter_non_objects_words(foundation_model, prompt, word_list) -> set:
    """
    Use LLM to filter out non-object words from a list.
    Returns a set of object words.
    """
    full_prompt = f'{prompt}\n\n{to_json_string(word_list)}'

    resp = post_with_retry(foundation_model, full_prompt)
    
    if resp is None:
        return set()
    
    words = parse_json(resp)
    
    if words is None:
        return set()
    
    return set(words) if isinstance(words, list) else set()


def get_word_list(corpus):
    words = []
    
    for text in corpus:
        words += re.findall(r"\b\w+\b", text)
        
    return words


def get_overused_objects(foundation_model, prompt, qa_by_frame):
    all_words = get_word_list(entry['question'] for qa in qa_by_frame.values() for entry in qa)
    unique_words = list(set(all_words))
    object_words = filter_non_objects_words(
        foundation_model = foundation_model,
        prompt = prompt,
        word_list = unique_words
    )
    
    if not object_words:
        logger.warning("Failed to found words related to object in filter_frequent_objects()")
        return None, None

    object_counts = Counter(word for word in all_words if word in object_words)

    median_count = np.median(list(object_counts.values()))
    threshold = median_count * 3
    overused_objects = {obj: cnt for obj, cnt in object_counts.items() if cnt > threshold}
    
    return overused_objects, threshold


def filter_frequent_objects(foundation_model, prompt, qa_by_frame):
    """
    Remove questions about objects that appear too frequently.
    Compute 3×median threshold and drop excess.
    """
    overused_objects, threshold = get_overused_objects(foundation_model, prompt, qa_by_frame)
    
    if overused_objects is None:
        return qa_by_frame
    
    all_qa_entries = [{'frame': frame, 'qa': qa} for frame, qa_list in qa_by_frame.items() for qa in qa_list]
    random.shuffle(all_qa_entries)

    filtered_qa = {}
    removed_count = 0
    for qa_entry in all_qa_entries:
        frame = qa_entry['frame']
        qa = qa_entry['qa']
        
        words = get_word_list([qa['question']])
        words_to_remove = [word for word in words if word in overused_objects]
        to_remove_counts = Counter(words_to_remove)
        
        if not to_remove_counts:
            filtered_qa[frame] = filtered_qa.get(frame, []) + [qa]
        else:
            removed_count += 1
            
            for word, cnt in to_remove_counts.items():
                overused_objects[word] = overused_objects[word] - cnt
                
                if overused_objects[word] <= threshold:
                    overused_objects.pop(word)
                
            if not overused_objects:
                break

    logger.info("Frequency filter removed %d questions", removed_count)

    return filtered_qa


def filter_duplicates_and_conflicts(qa_by_frame: dict) -> dict:
    """
    Remove exact duplicates and resolve conflicts:
      - Prefer 'Yes' over 'No' for boolean
      - For numeric, keep the highest
    """
    all_questions = {}
    
    for frame, qa_list in qa_by_frame.items():
        for qa in qa_list:
            question = qa["question"]
            answer   = qa["answer"]
            q_type   = infer_answer_type(answer)
            
            all_questions[question] = \
                all_questions.get(question, []) + [{'answer': answer, 'frame': frame, 'type': q_type, 'qa': qa}]
                
    filtered = {frame: [] for frame in qa_by_frame.keys()}
    
    for answers in all_questions.values():
        boolean = [entry for entry in answers if entry['type'] == 'boolean']
        numeric = [entry for entry in answers if entry['type'] == 'numeric']
        textual = [entry for entry in answers if entry['type'] == 'text']
        
        if boolean:
            entry = max(boolean, key=lambda x: x['answer'].lower() == 'yes') # prefer yes
            filtered[entry['frame']].append(entry['qa'])
            
        if numeric:
            entry = max(numeric, key=lambda x: int(x['answer'])) # prefer bigger
            filtered[entry['frame']].append(entry['qa'])
            
        if textual:
            for entry in textual:
                filtered[entry['frame']].append(entry['qa'])

    return filtered


def remove_wrong_measurement_questions(qa_by_frame: dict,
                                       scene_counts: Counter,
                                       vqa_dir: str) -> dict:
    """
    Drop Measurement questions whose numeric answer mismatches the scene_counts.
    """   
    filtered = {}

    for frame, qa_list in qa_by_frame.items():
        new_qa_list = []
        
        for qa in qa_list:
            if qa.get("category") == "Measurement":
                try:
                    num = int(qa["answer"].replace(",", ""))
                except:
                    new_qa_list.append(qa)
                    continue
                
                pattern = re.compile(r"How many ([\w\s\-]+?) (?:are|is) ", re.I)
                matches = pattern.match(qa["question"])
                
                if matches:
                    obj = matches.group(1).strip().lower()
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
                    
            new_qa_list.append(qa)
            
        filtered[frame] = new_qa_list

    return filtered


def iterative_neural_validation(
    foundation_model, 
    prompt, 
    qa_by_frame: dict,
    scene_description: str,
    max_iterations: int = 5
) -> dict:
    """
    Iteratively call validate_qa() until no changes or max_iterations reached.
    """
    current_qa = deepcopy(qa_by_frame)

    for i in range(max_iterations):
        logger.info("Neural validation iteration %d/%d", i+1, max_iterations)
        
        updated_qa = {}
        any_changed = False
        total_before = sum(len(value) for value in current_qa.values())

        for frame, qa_list in tqdm(current_qa.items(), desc="Validating QA"):
            raw_results = validate_qa(
                foundation_model, prompt, frame, qa_list, scene_description
            )

            validated = [
                qa for qa in raw_results
                if isinstance(qa, dict) and all(key in qa for key in ['question', 'answer', 'category'])
            ]

            original = [
                qa for qa in qa_list
                if isinstance(qa, dict) and all(key in qa for key in ['question', 'answer', 'category'])
            ]

            set_validated = {(qa["question"], qa["answer"]) for qa in validated}
            set_original  = {(qa["question"], qa["answer"]) for qa in original}
            
            if set_validated != set_original:
                any_changed = True

            updated_qa[frame] = validated if validated else qa_list

        total_after = sum(len(value) for value in updated_qa.values())
        removed = total_before - total_after
        logger.info("Iteration %d: removed %d questions", i+1, removed)
        
        if not any_changed:
            break

        current_qa = updated_qa

    return current_qa

