"""
Re-export Configuration from utils
"""
from src.utils.config_loader import Configuration

# Alias for backward compatibility
__all__ = ["Configuration"]

if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.config <yaml_path>")
        sys.exit(1)

    yaml_path = sys.argv[1]
    conf = Configuration(yaml_path=yaml_path)
    print(conf)
