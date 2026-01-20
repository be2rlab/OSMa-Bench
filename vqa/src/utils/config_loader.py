"""
Configuration loader for YAML files.
"""
import yaml
import os

class Configuration:
    """
    Loads configuration from a YAML file or dictionary entries.
    """
    def __init__(self, yaml_path: str = None, **entries):
        if yaml_path:
            if not os.path.exists(yaml_path):
                raise FileNotFoundError(f"YAML file '{yaml_path}' not found.")
            with open(yaml_path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
            self._load_data(data)
        else:
            self._load_data(entries)

    def _load_data(self, data: dict):
        for key, value in data.items():
            # If the value is a nested dict, convert to Configuration
            if isinstance(value, dict):
                value = Configuration(**value)
            setattr(self, key, value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__})"
