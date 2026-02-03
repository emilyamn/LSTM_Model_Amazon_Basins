"""
Utilitários para manipulação de dados.
"""

import torch
import numpy as np
from typing import List
from src.data.data_structures import Sample


def custom_collate_fn(batch: List[Sample]) -> Sample:
    """
    Função collate personalizada para objetos Sample.
    
    Args:
        batch: Lista de objetos Sample
        
    Returns:
        Objeto Sample com batch colado
    """
    if isinstance(batch[0], Sample):
        return Sample(
            encoder_dyn=torch.stack([item.encoder_dyn for item in batch]),
            decoder_dyn=torch.stack([item.decoder_dyn for item in batch]),
            static=torch.stack([item.static for item in batch]),
            temporal_enc=torch.stack([item.temporal_enc for item in batch]),
            temporal_dec=torch.stack([item.temporal_dec for item in batch]),
            target=torch.stack([item.target for item in batch]),
            mask_enc=torch.stack([item.mask_enc for item in batch]),
            mask_dec=torch.stack([item.mask_dec for item in batch]),
            baseline_last=torch.stack([item.baseline_last for item in batch]),
            forecast_date=np.array([item.forecast_date for item in batch]),
            date_index=np.array([item.date_index for item in batch]),
        )
    else:
        return torch.utils.data.default_collate(batch)


def move_sample_to_device(sample: Sample, device: str) -> Sample:
    """
    Move um objeto Sample para o dispositivo especificado.
    
    Args:
        sample: Objeto Sample
        device: Dispositivo (cpu, cuda)
        
    Returns:
        Sample movido para o dispositivo
    """
    return Sample(
        encoder_dyn=sample.encoder_dyn.to(device) if isinstance(sample.encoder_dyn, torch.Tensor) else sample.encoder_dyn,
        decoder_dyn=sample.decoder_dyn.to(device) if isinstance(sample.decoder_dyn, torch.Tensor) else sample.decoder_dyn,
        static=sample.static.to(device) if isinstance(sample.static, torch.Tensor) else sample.static,
        temporal_enc=sample.temporal_enc.to(device) if isinstance(sample.temporal_enc, torch.Tensor) else sample.temporal_enc,
        temporal_dec=sample.temporal_dec.to(device) if isinstance(sample.temporal_dec, torch.Tensor) else sample.temporal_dec,
        target=sample.target.to(device) if isinstance(sample.target, torch.Tensor) else sample.target,
        mask_enc=sample.mask_enc.to(device) if isinstance(sample.mask_enc, torch.Tensor) else sample.mask_enc,
        mask_dec=sample.mask_dec.to(device) if isinstance(sample.mask_dec, torch.Tensor) else sample.mask_dec,
        baseline_last=sample.baseline_last.to(device) if isinstance(sample.baseline_last, torch.Tensor) else sample.baseline_last,
        forecast_date=sample.forecast_date,  # mantém numpy array
        date_index=sample.date_index,        # mantém numpy array
    )