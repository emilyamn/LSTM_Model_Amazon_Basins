"""
Utilitários para carregar configurações YAML.
"""

import yaml
from typing import Dict, Any, Union
import pathlib


def load_config(config_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """
    Carrega configurações de um arquivo YAML.
    
    Args:
        config_path: Caminho para o arquivo YAML
        
    Returns:
        Dicionário com configurações
    """
    config_path = pathlib.Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_all_configs(config_dir: Union[str, pathlib.Path]) -> Dict[str, Dict[str, Any]]:
    """
    Carrega todas as configurações de um diretório.
    
    Args:
        config_dir: Diretório com arquivos YAML
        
    Returns:
        Dicionário com todas as configurações
    """
    config_dir = pathlib.Path(config_dir)
    configs = {}
    
    for config_file in config_dir.glob("*.yaml"):
        config_name = config_file.stem  # Remove extensão .yaml
        configs[config_name] = load_config(config_file)
    
    return configs


def get_device(config: Dict[str, Any]) -> str:
    """
    Obtém dispositivo (CPU/GPU) baseado na configuração.
    
    Args:
        config: Dicionário de configuração
        
    Returns:
        String do dispositivo
    """
    import torch
    
    device_config = config.get('misc', {}).get('device', 'auto')
    
    if device_config == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device_config in ['cuda', 'gpu']:
        if not torch.cuda.is_available():
            print("⚠️ CUDA não disponível, usando CPU")
            return 'cpu'
        return 'cuda'
    else:
        return 'cpu'