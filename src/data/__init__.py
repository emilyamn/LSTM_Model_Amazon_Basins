"""
Módulo de dados e dataset.
"""

from .data_structures import Sample, Scaler, compute_scaler
from .dataset import (
    HydroDataset,
    create_temporal_split_with_gap,
    create_dataset_for_training
)

__all__ = [
    'Sample',
    'Scaler',
    'compute_scaler',
    'HydroDataset',
    'create_temporal_split_with_gap',
    'create_dataset_for_training',
]