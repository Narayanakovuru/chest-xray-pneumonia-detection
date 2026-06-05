import yaml
from pathlib import Path
from typing import Dict, Any, List

def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load and parse a YAML file.

    Args:
        file_path (Path): Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
    with open(file_path, "r") as f:
        try:
            content = yaml.safe_load(f)
            return content if content is not None else {}
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file {file_path}: {e}")

def load_all_configs(config_dir: str = "configs") -> Dict[str, Any]:
    """Load and merge configs from data.yaml, model.yaml, train.yaml, and inference.yaml.

    Args:
        config_dir (str): Directory containing the configuration files.

    Returns:
        Dict[str, Any]: A merged dictionary containing configurations.
    """
    base_path = Path(config_dir)
    config_files = ["data.yaml", "model.yaml", "train.yaml", "inference.yaml"]
    
    merged_config = {}
    for filename in config_files:
        path = base_path / filename
        file_config = load_yaml(path)
        merged_config.update(file_config)
        
    return merged_config
