"""
Utilitários para cálculo de eixos temporais.
"""

import numpy as np
from typing import Tuple, Dict, List
import torch

def compute_time_axes(
    flow_window_config: Dict,
    climate_window_config: Dict
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Calcula eixos temporais para encoder e decoder.
    
    Args:
        flow_window_config: Configuração de janelas para fluxo
        climate_window_config: Configuração de janelas para clima
        
    Returns:
        Tuple com (encoder_offsets, decoder_offsets, decoder_history, decoder_horizon)
    """
    # Encoder (considera travel_time)
    enc_min_flow = min(
        spec["start"] - spec.get("travel_time", 0)
        for st in flow_window_config
        for spec in flow_window_config[st]["encoder"]
    )
    enc_min_clim = min(cfg["encoder"][0] for cfg in climate_window_config.values())
    enc_min = min(enc_min_flow, enc_min_clim)
    enc_max = -1

    # Decoder (início travel-aware; fim pelo maior end)
    dec_min_flow = min(
        spec["start"] - spec.get("travel_time", 0)
        for st in flow_window_config
        for spec in flow_window_config[st]["decoder"]
    )
    dec_min_clim = min(cfg["decoder"][0] for cfg in climate_window_config.values())
    dec_min = min(dec_min_flow, dec_min_clim)

    dec_max_flow = max(spec["end"] for st in flow_window_config 
                      for spec in flow_window_config[st]["decoder"])
    dec_max_clim = max(cfg["decoder"][1] for cfg in climate_window_config.values())
    dec_max = max(dec_max_flow, dec_max_clim)

    encoder_offsets = np.arange(enc_min, enc_max + 1)
    decoder_offsets = np.arange(dec_min, dec_max + 1)
    decoder_history = np.sum(decoder_offsets < 0)
    decoder_horizon = np.sum(decoder_offsets >= 0)
   
    return encoder_offsets, decoder_offsets, decoder_history, decoder_horizon