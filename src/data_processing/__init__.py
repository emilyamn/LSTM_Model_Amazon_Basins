"""
Módulo de processamento de dados hidrológicos.

Inclui:
- Pré-processamento de séries temporais
- Interpolação espacial (IDW)
- Geração de dados de forecast
- Feature engineering para treino
- Processamento de inferência em produção
"""

# Complete Series
from .complete_series import DataPreprocessor

# Interpolação
from .interpolate_series import (
    load_station_data,
    calculate_idw_weights,
    interpolate_and_overwrite,
    batch_interpolate_and_overwrite
)

# Forecast
from .working_with_forecast import (
    ForecastGenerator,
    generate_forecast_files
)

# Feature Engineering (Treino)
from .features_processing import (
    HydroFeatureEngineer,
    ForcingType,
    load_station_data as load_observed_data,
    load_forecast_data,
    merge_observed_and_forecast,
    process_features
)

# Feature Engineering (Inferência/Produção)
from .features_processing_inference import (
    InferenceConfig,
    process_inference,
    INTERNAL_COLUMN_NAMES
)

__all__ = [
    # Complete Series
    'DataPreprocessor',
    
    # Interpolação
    'load_station_data',
    'calculate_idw_weights',
    'interpolate_and_overwrite',
    'batch_interpolate_and_overwrite',
    
    # Forecast
    'ForecastGenerator',
    'generate_forecast_files',
    
    # Feature Engineering - Treino
    'HydroFeatureEngineer',
    'ForcingType',
    'load_observed_data',
    'load_forecast_data',
    'merge_observed_and_forecast',
    'process_features',
    
    # Feature Engineering - Inferência
    'InferenceConfig',
    'process_inference',
    'INTERNAL_COLUMN_NAMES',
]
