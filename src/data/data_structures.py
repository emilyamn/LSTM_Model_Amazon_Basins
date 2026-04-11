"""
Estruturas de dados para o dataset hidrológico.
"""

from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np


@dataclass
class Scaler:
    """Classe para escalonamento de dados."""
    mean: float
    std: float

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Aplica transformação de escalonamento."""
        return (values - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Aplica transformação inversa de escalonamento."""
        return values * (self.std + 1e-8) + self.mean


@dataclass
class Sample:
    """Amostra do dataset hidrológico."""
    encoder_dyn: torch.Tensor
    decoder_dyn: torch.Tensor
    static: torch.Tensor
    temporal_enc: torch.Tensor
    temporal_dec: torch.Tensor
    target: torch.Tensor
    mask_enc: torch.Tensor
    mask_dec: torch.Tensor
    baseline_last: torch.Tensor
    forecast_date: Optional[np.ndarray] = None  # data do primeiro passo de previsão
    date_index: Optional[int] = None           # índice do center no dataframe original


def compute_scaler(values: np.ndarray, silent: bool = False) -> Scaler:
    """
    Calcula scaler para um conjunto de valores.

    Args:
        values: Array com valores
        silent: Se True, suprime warnings (usado para scalers dummy)

    Returns:
        Objeto Scaler
    """
    # Remove NaN antes de calcular estatísticas
    clean_values = values[~np.isnan(values)]

    if len(clean_values) == 0:
        if not silent:
            print("⚠️ Aviso: Todos os valores são NaN, usando scaler padrão")
        return Scaler(mean=0.0, std=1.0)

    mean = float(np.mean(clean_values))
    std = float(np.std(clean_values))

    if std < 1e-6:
        if not silent:
            print(f"⚠️ Aviso: Desvio padrão muito baixo ({std}), ajustando para 1.0")
        std = 1.0

    return Scaler(mean, std)