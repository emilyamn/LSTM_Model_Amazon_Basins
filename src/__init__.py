"""
Módulo principal para previsão hidrológica.
"""
from .data_processing import (
    # Complete Series
    DataPreprocessor,

    # Interpolação
    batch_interpolate_and_overwrite,

    # Forecast
    ForecastGenerator,
    generate_forecast_files,

    # Feature Engineering - Treino
    HydroFeatureEngineer,
    ForcingType,
    load_observed_data,
    load_forecast_data,
    merge_observed_and_forecast,
    process_features,

    # Feature Engineering - Inferência
    InferenceConfig,
    process_inference,
    INTERNAL_COLUMN_NAMES,
)

# Importações principais - Data
from .data.data_structures import Scaler, Sample
from .data.dataset import HydroDataset, create_temporal_split_with_gap, create_dataset_for_training_validation, create_dataset_for_inference

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

from .utils.data_utils import (
    custom_collate_fn,
    move_sample_to_device,
    get_device
)

from .utils.time_utils import (
    compute_time_axes
)

from .utils.serialization import (
    save_checkpoint,
    load_checkpoint,
    load_checkpoint_legacy,
    load_checkpoint_legacy_with_climate
)

from .utils.experiment_utils import (
    convert_predictions_to_df,
    create_experiment,
    load_experiment,
    save_model,
    save_predictions,
    save_metrics,
    save_plot,
    list_experiments,
    print_experiment_summary,
    get_experiment_path,
    find_experiment_by_name
)

# Importações principais - Result Analysis
from .result_analysis.extract_flow_extremes import (
    analyze_flow_extremes,
)

from .result_analysis.metrics import (
    compute_flow_metrics,
    print_metrics_summary,
    compute_metrics_by_event_type,
    print_metrics_comparison_by_event,
)

from .result_analysis.plots import (
    plot_predictions_with_context,
    plot_metrics_by_horizon,
    plot_full_series_with_d1_forecast,
    plot_predictions_extremes,
    plot_metrics_by_horizon_comparison,
    plot_forecast_horizons_analysis
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
    # ===== DATA PROCESSING =====
    "DataPreprocessor",
    "batch_interpolate_and_overwrite",
    "ForecastGenerator",
    "generate_forecast_files",
    
    # Feature Engineering
    "HydroFeatureEngineer",
    "ForcingType",  # ← ADICIONAR
    "load_observed_data",
    "load_forecast_data",
    "merge_observed_and_forecast",
    "process_features",
    
    # Inferência
    "InferenceConfig",
    "process_inference",
    "INTERNAL_COLUMN_NAMES",
    
    # ===== DATA STRUCTURES & DATASET =====
    "Scaler",
    "Sample",
    "HydroDataset",
    "create_temporal_split_with_gap",
    "create_dataset_for_training_validation",
    "create_dataset_for_inference",
    
    # ===== MODEL =====
    "Seq2SeqHydro",
    "StaticEmbedding",
    
    # ===== TRAINING =====
    "train_model",
    "predict_autoregressive",
    "multi_step_loss",
    
    # ===== UTILS =====
    "convert_predictions_to_df",
    "ConfigLoader",
    "load_feature_config",
    "load_split_config",
    "load_config",
    "load_all_configs",
    "get_device",
    "custom_collate_fn",
    "move_sample_to_device",
    "compute_time_axes",
    "save_checkpoint",
    "load_checkpoint",
    "create_experiment",
    "load_experiment",
    "save_model",
    "save_predictions",
    "save_metrics",
    "save_plot",
    "list_experiments",
    "print_experiment_summary",
    "get_experiment_path",
    "find_experiment_by_name",
    
    # ===== RESULT ANALYSIS =====
    "compute_flow_metrics",
    "print_metrics_summary",
    "plot_predictions_with_context",
    "plot_metrics_by_horizon",
    "plot_full_series_with_d1_forecast",
    "analyze_flow_extremes",
    "compute_metrics_by_event_type",
    "print_metrics_comparison_by_event",
    "plot_forecast_horizons_analysis",
    
    # ===== LINAR =====
    "interpolate_linar",
    "interpolate_linear",
    "convert_to_series",
    "resample_timeseries",
    "group_nans",
]