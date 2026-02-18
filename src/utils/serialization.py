"""
Utilitários para salvar e carregar checkpoints do modelo e metadados.
"""

from typing import Dict, Any, Tuple
import torch
from src.model.architecture import Seq2SeqHydro

def save_checkpoint(
    model: Seq2SeqHydro,
    inference_meta: Dict[str, Any],
    model_config: Dict[str, Any],
    path: str
):
    """
    Salva o modelo treinado junto com metadados essenciais para inferência.

    Args:
        model: Instância do modelo treinado
        inference_meta: Dicionário com scalers e configs do dataset
        model_config: Hiperparâmetros usados para criar o modelo
        path: Caminho para salvar o arquivo (.pth ou .pkl)
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "inference_meta": inference_meta,
    }
    torch.save(checkpoint, path)
    print(f"✅ Checkpoint salvo com sucesso em: {path}")

def load_checkpoint(path: str, device: str = "cpu") -> Tuple[Seq2SeqHydro, Dict[str, Any]]:
    """
    Carrega um checkpoint e reconstrói o modelo.

    Returns:
        model: Modelo carregado e em modo eval()
        inference_meta: Metadados (scalers, configs, etc.)
    """
    checkpoint = torch.load(path, map_location=device)

    # Reconstruir arquitetura
    config = checkpoint["model_config"]
    model = Seq2SeqHydro(**config)

    # Carregar pesos
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint["inference_meta"]