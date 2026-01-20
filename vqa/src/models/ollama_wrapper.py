import ollama

from src.models.utils import encode_image

class OllamaModel:
    def __init__(self, model=None, llm=None, lvlm=None, **kwargs):
        if llm is not None and lvlm is not None:
            self.llm = llm
            self.lvlm = lvlm
        elif model is not None:
            self.llm = model
            self.lvlm = model
        else:
            raise ValueError("model or llm and lvlm must not be None")
    
    def generate(self, prompt=None, images=None):
        if images is not None and not isinstance(images, list):
            images = [images]
            
        args_dict = {
            "model": self.llm,
            "prompt": prompt,
            "stream": False
        }
        
        if images is not None:
            args_dict['images'] = [encode_image(img) for img in images]
            args_dict['model'] = self.lvlm
            
        try:
            response = ollama.generate(**args_dict)
            
            return response['response']
        except Exception as e:
            print(e)
            return None
            
if __name__ == "__main__":
    model = OllamaModel(model='llava:7b')
    
    answer = model.generate(prompt='Tell me a joke, please')
    print(answer)