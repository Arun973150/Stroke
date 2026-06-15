"""
Config Loader Utility
=====================
Handles YAML configuration loading and validation for Sentinel Stroke pipeline.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        # Default config location
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "nnunet_config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    _validate_config(config)
    
    # Expand environment variables in paths
    config = _expand_paths(config)
    
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate that required configuration fields exist."""
    required_sections = ['paths', 'dataset', 'preprocessing', 'training']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate paths
    required_paths = ['raw_data', 'preprocessed', 'nnunet_raw', 
                      'nnunet_preprocessed', 'nnunet_results']
    for path_name in required_paths:
        if path_name not in config['paths']:
            raise ValueError(f"Missing required path: {path_name}")


def _expand_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand environment variables and ~ in paths."""
    if 'paths' in config:
        for key, value in config['paths'].items():
            if isinstance(value, str):
                config['paths'][key] = os.path.expanduser(os.path.expandvars(value))
    return config


def setup_nnunet_env(config: Dict[str, Any]) -> None:
    """
    Set up nnU-Net environment variables from config.
    
    Args:
        config: Configuration dictionary
    """
    os.environ['nnUNet_raw'] = config['paths']['nnunet_raw']
    os.environ['nnUNet_preprocessed'] = config['paths']['nnunet_preprocessed']
    os.environ['nnUNet_results'] = config['paths']['nnunet_results']
    
    # Create directories if they don't exist
    for path_key in ['nnunet_raw', 'nnunet_preprocessed', 'nnunet_results', 
                     'logs', 'checkpoints']:
        if path_key in config['paths']:
            Path(config['paths'][path_key]).mkdir(parents=True, exist_ok=True)


def get_dataset_name(config: Dict[str, Any]) -> str:
    """Get full dataset name in nnU-Net format."""
    dataset_id = config['dataset']['id']
    dataset_name = config['dataset']['name']
    return f"Dataset{dataset_id}_{dataset_name}"


def print_config(config: Dict[str, Any]) -> None:
    """Pretty print configuration."""
    print("=" * 70)
    print("Configuration")
    print("=" * 70)
    
    print("\nPaths:")
    for key, value in config['paths'].items():
        print(f"  {key}: {value}")
    
    print(f"\nDataset:")
    print(f"  ID: {config['dataset']['id']}")
    print(f"  Name: {config['dataset']['name']}")
    
    print(f"\nPreprocessing:")
    print(f"  Target spacing: {config['preprocessing']['target_spacing']}")
    print(f"  Train/Val/Test: {config['preprocessing']['train_ratio']:.0%}/"
          f"{config['preprocessing']['val_ratio']:.0%}/"
          f"{config['preprocessing']['test_ratio']:.0%}")
    
    print(f"\nTraining:")
    print(f"  Configuration: {config['training']['configuration']}")
    print(f"  Folds: {config['training']['folds']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Test config loading
    config = load_config()
    print_config(config)
