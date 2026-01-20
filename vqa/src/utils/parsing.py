import json
import re
from collections import Counter

def extract_single_category_prompt(full_prompt: str, target_category: str) -> str:
    """
    Extract the JSON block from the full prompt and keep only the target category.
    """
    try:
        start = full_prompt.find('{')
        end = full_prompt.rfind('}')
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Cannot locate JSON block in prompt.")
        json_str = full_prompt[start:end+1]
        parsed = json.loads(json_str)
        categories = parsed.get("categories", [])
        selected = [c for c in categories if c.get("name") == target_category]
        if not selected:
            raise ValueError(f"Category '{target_category}' not found.")
        parsed["categories"] = selected
        return json.dumps(parsed, indent=4)
    except Exception as e:
        print(f"[!] Error parsing qa_generation_prompt: {e}")
        raise

# mapping English numerals to digits
DIGIT_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3",  "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10"
}

# regex to catch too-generic spatial relations
GENERIC_SPATIAL = re.compile(r"\b(on|at|in|on the|on a) (wall|floor|ceiling)\b", re.I)

def is_yes_no_answer(answer: str) -> bool:
    """
    Return True if the answer is exactly 'Yes' or 'No' (case-insensitive).
    """
    return str(answer).strip().lower() in {"yes", "no"}

def is_numeric_answer(answer: str) -> bool:
    """
    Return True if the answer is a pure integer or float string.
    """
    return re.match(r"^\d+(\.\d+)?$", str(answer).strip()) is not None

def infer_answer_type(answer: str) -> str:
    """
    Infer answer type: 'boolean', 'numeric', or 'text'.
    """
    a = str(answer).strip().lower()
    if a in {"yes", "no"}:
        return "boolean"
    if is_numeric_answer(a):
        return "numeric"
    return "text"

def build_scene_inventory(desc_dict: dict) -> Counter:
    """
    Count approximate number of objects across all frames.
    desc_dict: mapping frame_name -> description_text.
    Returns a Counter { object_name: total_count }.
    """
    inv = Counter()
    # pattern matches digits or words from DIGIT_MAP followed by an object name
    pattern = re.compile(
        rf"\b(\d+|{'|'.join(DIGIT_MAP.keys())})\s+([\w\-]+)\b", re.I
    )
    for text in desc_dict.values():
        for num, obj in pattern.findall(text):
            if num.isdigit():
                n = int(num)
            else:
                n = int(DIGIT_MAP[num.lower()])
            inv[obj.lower()] += n
    return inv
