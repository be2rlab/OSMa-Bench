import json
import re
from typing import Any, Optional, List, Dict

def load_json(filepath):
    """
    Load a JSON file and return the parsed object.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filepath):
    """
    Save an object as JSON to the given filepath.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def clean_json_response(response_text):
    """
    Remove Markdown/fenced code blocks around JSON and return cleaned text.
    """
    # Strip ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"^```json", "", response_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"^```", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)
    return cleaned.strip()


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
    return questions

def to_json_string(obj: Any, indent: int = 2) -> str:
    """
    Pretty-print Python object as JSON string.
    """
    return json.dumps(obj, indent=indent)

def parse_json(text: str) -> Optional[Any]:
    """
    Clean fenced JSON from LLM and parse it.
    Returns Python object or None on failure.
    """
    from .json_utils import clean_json_response  # avoid circular import
    cleaned = clean_json_response(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None