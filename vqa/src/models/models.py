def create_foundation_model(api_name, **kwargs):
    if api_name == 'ollama':
        from src.models.ollama_wrapper import OllamaModel
        return OllamaModel(**kwargs)
    elif api_name == 'gemini':       
        from src.models.gemini_wrapper import GeminiModel
        return GeminiModel(**kwargs)
    else:
        raise ValueError(f"Unsupported api: {api_name}")