import yaml
from pydantic import BaseModel
from pathlib import Path


class FoundationModel(BaseModel):
    api_name: str
    api_key: str = None
    model: str = None
    llm: str = None
    lvlm: str = None


class DescriptionGeneration(BaseModel):
    prompt: str
    rejection_keyword: str
    pos_threshold: float = 1.5
    angle_threshold: float = 10.0
    
    
class QAGeneration(BaseModel):
    prompt: str


class QAValidation(BaseModel):
    neural_val_prompt: str
    filter_non_objects_prompt: str


class CGAnswering(BaseModel):
    prompt: str
    batch_size: int = 10
    
class Evaluation(BaseModel):
    prompt: str
    batch_size: int = 10



class Configuration(BaseModel):
    foundation_model: FoundationModel
    
    description_generation: DescriptionGeneration
    qa_generation: QAGeneration
    qa_validation: QAValidation

    cg_answering: CGAnswering
    
    evaluation: Evaluation
    
    data_dir: Path
    output_dir: Path


def load_configuration(yaml_path, **entries):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
        
    data.update(entries)

    config = Configuration(**data)
    
    return config
