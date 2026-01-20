import time
import requests

def post_with_retry(url, payload, max_retries=5, delay_seconds=10):
    """
    Post JSON payload to URL with retries on 429/503 status.
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload)
        except Exception as e:
            print(f"Request exception: {e}")
            time.sleep(delay_seconds)
            continue

        if response.status_code == 200:
            return response
        if response.status_code in (429, 503):
            print(f"Server busy (status {response.status_code}), retrying in {delay_seconds} seconds "
                  f"(attempt {attempt+1}/{max_retries})...")
            time.sleep(delay_seconds)
            continue

        print(f"Error: {response.status_code} - {response.text}")
        return None

    print("Max retries exceeded.")
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
