"""
Módulo principal para previsão hidrológica.
"""

# Importações principais - Data
from .data.data_structures import Scaler, Sample
from .data.dataset import HydroDataset, create_temporal_split_with_gap

# Importações principais - Model
from .model.architecture import Seq2SeqHydro
from .model.layers import StaticEmbedding

# Importações principais - Training
from .training.trainer import train_model, predict_autoregressive
from .training.losses import multi_step_loss

# Importações principais - Utils
from .utils.config_loader import (
    ConfigLoader, 
    load_feature_config, 
    load_split_config,
    load_config,
    load_all_configs
)
from .utils.data_utils import custom_collate_fn, move_sample_to_device, get_device
from .utils.time_utils import compute_time_axes
from .utils.serialization import save_checkpoint, load_checkpoint

# Importações principais - Result Analysis
from .result_analysis import (
    compute_flow_metrics,
    print_metrics_summary,
    plot_predictions_with_context,
    plot_metrics_by_horizon
)

# Importações principais - LinAR
from .linar import (
    interpolate_linar,
    interpolate_linear,
    convert_to_series,
    resample_timeseries,
    group_nans
)

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
    # Utils - Serialization
    "save_checkpoint", "load_checkpoint",
    # Result Analysis
    "compute_flow_metrics",
    "print_metrics_summary",
    "plot_predictions_with_context",
    "plot_metrics_by_horizon",
    # LinAR Interpolation
    "interpolate_linar",
    "interpolate_linear",
    "convert_to_series",
    "resample_timeseries",
    "group_nans",
]