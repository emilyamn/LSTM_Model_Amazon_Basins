"""
Utilitários para cálculo de eixos temporais.
"""

import numpy as np
from typing import Tuple, Dict


def compute_time_axes(
    flow_window_config: Dict,
    climate_window_config: Dict,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Calcula eixos temporais para encoder e decoder.

    Ambas as configurações usam um único par encoder/decoder aplicado a todas
    as estações (sem hierarquia espacial, sem travel_time):

        flow_window_config = {
            "encoder": {"start": -30, "end": -1},
            "decoder": {"start": -30, "end": -1},
        }
        climate_window_config = {
            "encoder": {"start": -30, "end": -1},
            "decoder": {"start": -30, "end": 14},
        }

    Returns:
        (encoder_offsets, decoder_offsets, decoder_history, decoder_horizon)
    """
    enc_min = min(
        flow_window_config["encoder"]["start"],
        climate_window_config["encoder"]["start"],
    )
    enc_max = max(
        flow_window_config["encoder"]["end"],
        climate_window_config["encoder"]["end"],
    )

    dec_min = min(
        flow_window_config["decoder"]["start"],
        climate_window_config["decoder"]["start"],
    )
    dec_max = max(
        flow_window_config["decoder"]["end"],
        climate_window_config["decoder"]["end"],
    )

    encoder_offsets = np.arange(enc_min, enc_max + 1)
    decoder_offsets = np.arange(dec_min, dec_max + 1)
    decoder_history = int(np.sum(decoder_offsets < 0))
    decoder_horizon = int(np.sum(decoder_offsets >= 0))

    return encoder_offsets, decoder_offsets, decoder_history, decoder_horizon
