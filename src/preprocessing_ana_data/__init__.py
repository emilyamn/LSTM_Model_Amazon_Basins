'''
Módulo de processamento de séries hidrológicas da ANA.
'''
from .ana_data_into_series import (
    process_vazao,
    process_cota,
    process_precipitacao,
    convert_all_ana_series,
)
from .merge_ana_data import build_raw_dataset

__all__ = [
    # processamento individual
    'process_vazao',
    'process_cota',
    'process_precipitacao',
    # pipeline completo — etapa 1
    'convert_all_ana_series',
    # pipeline completo — etapa 2
    'build_raw_dataset',
]