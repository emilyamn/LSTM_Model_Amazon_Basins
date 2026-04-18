"""
Módulo para análise de resultados do modelo.
"""

from .metrics import (compute_flow_metrics,
    print_metrics_summary,
    compute_metrics_by_event_type,
    print_metrics_comparison_by_event
)
from .plots import (
    plot_predictions_with_context,
    plot_metrics_by_horizon,
    plot_full_series_with_d1_forecast,
    plot_predictions_extremes,
    plot_metrics_by_horizon_comparison,
    plot_forecast_horizons_analysis
)
from .extract_flow_extremes import analyze_flow_extremes

__all__ = [
    'compute_flow_metrics',
    'print_metrics_summary',
    'compute_metrics_by_event_type',
    'print_metrics_comparison_by_event',
    'plot_predictions_with_context',
    'plot_metrics_by_horizon',
    'plot_full_series_with_d1_forecast',
    'analyze_flow_extremes',
    'plot_predictions_extremes',
    'plot_metrics_by_horizon_comparison',
    'plot_forecast_horizons_analysis'
]