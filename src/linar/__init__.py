"""
Módulo LinAR para interpolação de séries temporais.

Implementação baseada no método LinAR (Linear AutoRegressive interpolation)
para preenchimento de lacunas em séries temporais hidrológicas.
"""

from .LinAR_functions import (
    convert_to_series,
    resample_timeseries,
    group_nans,
    fill_data,
    update_timeseries,
    difference,
    f_test,
    get_stationary_data,
    get_trend_and_breakpoints,
    create_model,
    undiff,
    get_undifferenced_data,
    adjust_to_next_obs,
    interpolate_linear,
    interpolate_linar
)

__all__ = [
    # Funções principais
    'interpolate_linar',
    'interpolate_linear',

    # Funções auxiliares de pré-processamento
    'convert_to_series',
    'resample_timeseries',
    'group_nans',

    # Funções de manipulação de dados
    'fill_data',
    'update_timeseries',

    # Funções de estacionariedade
    'difference',
    'f_test',
    'get_stationary_data',

    # Funções de modelagem AR
    'get_trend_and_breakpoints',
    'create_model',
    'undiff',
    'get_undifferenced_data',
    'adjust_to_next_obs',
]