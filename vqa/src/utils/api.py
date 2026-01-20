import logging
import time

logger = logging.getLogger(__name__)

def post_with_retry(model, prompt, images=None, max_retries=5, delay_seconds=2):
    for _ in range(max_retries):
        answer = model.generate(prompt=prompt, images=images)
        
        if answer is not None:
            return answer
        
        time.sleep(delay_seconds)
        
    logger.error(f"Content generation error: maximum retries {max_retries} reached for post_with_retry")
        
    return None


def request_gemini(config, prompt, use_llm=False):
    """
    Send a request to Gemini API (VLM or LLM) and return text content.
    """
    model = config.llm if use_llm else config.vlm
    api_url = f"{config.url}/{model}:generateContent?key={config.gemini_api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    response = post_with_retry(api_url, payload)
    if response is None:
        return None

    candidates = response.json().get("candidates", [])
    if not candidates:
        return None

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return None

    text = parts[0].get("text", "")
    # Strip code fences if present
    return text.strip().lstrip("```json").rstrip("```").strip()
