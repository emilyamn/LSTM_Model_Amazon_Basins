"""
Utilitários para salvar e carregar checkpoints do modelo e metadados.
"""

from typing import Dict, Any, Tuple
import torch
import pickle
from src.model.architecture import Seq2SeqHydro
from src.data.data_structures import Scaler

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
    # Tenta carregar de forma segura primeiro (PyTorch 2.6+)
    try:
        # Adiciona Scaler à lista de globais seguros
        with torch.serialization.safe_globals([Scaler]):
            checkpoint = torch.load(path, map_location=device, weights_only=True)
    except (AttributeError, TypeError, pickle.UnpicklingError, RuntimeError) as e:
        # Se falhar (por exemplo, numpy scalars não permitidos ou versão antiga),
        # faz fallback para o método padrão (weights_only=False)
        print(f"⚠️ Aviso: Carregamento seguro falhou ({str(e)}).")
        print("🔄 Tentando carregar com weights_only=False (necessário para checkpoints antigos ou com tipos complexos)...")
        checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Reconstruir arquitetura
    config = checkpoint["model_config"]
    model = Seq2SeqHydro(**config)

    # Carregar pesos
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint["inference_meta"]


def load_checkpoint_legacy(path: str, device: str = "cpu") -> Tuple[Seq2SeqHydro, Dict[str, Any]]:
    """
    Carrega checkpoints antigos que podem ter arquitetura diferente da atual.

    Lida com:
      - Chaves extras no state_dict (ex: climate_proj removido)
      - Parâmetros faltantes no model_config (ex: n_decoder_flow_feats)

    Returns:
        model: Modelo carregado e em modo eval()
        inference_meta: Metadados (scalers, configs, etc.)
    """
    try:
        with torch.serialization.safe_globals([Scaler]):
            checkpoint = torch.load(path, map_location=device, weights_only=True)
    except (AttributeError, TypeError, pickle.UnpicklingError, RuntimeError):
        checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = checkpoint["model_config"]

    # Garantir que n_decoder_flow_feats existe (checkpoints antigos não têm)
    config.setdefault("n_decoder_flow_feats", 0)

    model = Seq2SeqHydro(**config)

    # Carregar pesos com tolerância a chaves extras/faltantes
    state_dict = checkpoint["model_state_dict"]
    model_keys = set(model.state_dict().keys())
    saved_keys = set(state_dict.keys())

    extra = saved_keys - model_keys
    missing = model_keys - saved_keys

    if extra:
        print(f"⚠️ Legacy: ignorando {len(extra)} chaves extras: {sorted(extra)[:5]}...")
    if missing:
        print(f"⚠️ Legacy: {len(missing)} chaves não encontradas (usam init padrão): {sorted(missing)[:5]}...")

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, checkpoint["inference_meta"]
