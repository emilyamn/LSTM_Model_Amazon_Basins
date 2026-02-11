"""
Módulo principal para previsão hidrológica.
"""

__version__ = "1.0.0"
__author__ = "Sistema de Previsão Hidrológica"

# Importações principais
from .data.data_structures import Scaler, Sample
from .data.dataset import HydroDataset, create_temporal_split_with_gap
from .model.architecture import Seq2SeqHydro
from .model.layers import StaticEmbedding
from .training.trainer import train_model, predict_autoregressive
from .training.losses import multi_step_loss
from .utils.config_loader import (
    ConfigLoader, 
    load_feature_config, 
    load_split_config,
    load_config,
    load_all_configs
)
from .utils.data_utils import custom_collate_fn, move_sample_to_device, get_device
from .utils.time_utils import compute_time_axes

__all__ = [
    # Data structures
    "Scaler", "Sample",
    # Dataset
    "HydroDataset", "create_temporal_split_with_gap",
    # Model
    "Seq2SeqHydro", "StaticEmbedding",
    # Training
    "train_model", "predict_autoregressive", "multi_step_loss",
    # Utils - Config
    "ConfigLoader", "load_feature_config", "load_split_config",
    "load_config", "load_all_configs",
    # Utils - Data
    "get_device", "custom_collate_fn", "move_sample_to_device",
    # Utils - Time
    "compute_time_axes",
]