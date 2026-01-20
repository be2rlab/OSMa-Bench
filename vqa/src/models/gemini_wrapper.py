from google import genai

from src.models.utils import encode_image

class GeminiModel:
    def __init__(self, model, api_key=None, **kwargs):
        self.model = model
        self.client = genai.Client(api_key=api_key)
    
    def generate(self, prompt, images=None):
        if images is not None and not isinstance(images, list):
            images = [images]
            
        if images is not None:
            contents = [prompt] + images
        else:
            contents = prompt
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
            )
                    
            return response.text
        except Exception as e:
            print(e)
            return None
            
if __name__ == "__main__":
    api_key = ""
    model = GeminiModel(model='gemini-2.0-flash', api_key=api_key)
    
    answer = model.generate(prompt='Tell me a joke, please')
    print(answer)