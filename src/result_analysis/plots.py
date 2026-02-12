"""
Módulo para visualização de resultados do modelo.
"""

from datetime import timedelta
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_predictions_with_context(
    preds: np.ndarray,
    obs: np.ndarray,
    stations: List[int],
    df: pd.DataFrame,
    forecast_dates: np.ndarray,
    n_samples: int = 3,
    context_days: int = 30,
    baseline_last: Optional[np.ndarray] = None,
    g_seq: Optional[np.ndarray] = None,
) -> None:
    """
    Plota previsões com contexto temporal (últimos N dias antes da previsão).

    Args:
        preds: Previsões, shape (batch, horizon, n_stations)
        obs: Observações, shape (batch, horizon, n_stations)
        stations: Lista de IDs das estações
        df: DataFrame original com índice temporal (colunas Q_{station_id})
        forecast_dates: Datas de início de cada previsão, shape (batch,)
        n_samples: Número de amostras aleatórias para plotar
        context_days: Quantos dias anteriores mostrar como contexto
        baseline_last: Baseline de persistência, shape (batch, n_stations)
        g_seq: Sequência de gates, shape (batch, horizon, n_stations)
    """
    B, T, S = preds.shape
    n_samples = min(n_samples, B)

    # Selecionar amostras aleatórias
    sample_indices = np.random.choice(B, n_samples, replace=False)

    for idx in sample_indices:
        forecast_start = pd.to_datetime(forecast_dates[idx])

        # Definir janela de contexto
        context_start = forecast_start - timedelta(days=context_days)
        #forecast_end = forecast_start + timedelta(days=T-1)

        fig, axes = plt.subplots(S, 1, figsize=(14, 4 * S), sharex=True)
        axes = np.atleast_1d(axes)
        twin_axes = []

        for st_i, st_id in enumerate(stations):
            ax = axes[st_i]

            # ==========================================
            # CONTEXTO: dados históricos
            # ==========================================
            context_mask = (df.index >= context_start) & (df.index < forecast_start)
            context_dates = df.index[context_mask]
            context_values = df[f"Q_{st_id}"].loc[context_mask].values

            if len(context_dates) > 0:
                ax.plot(context_dates, context_values,
                       color='gray', linewidth=1.5, alpha=0.6,
                       label='Histórico (contexto)')

            # ==========================================
            # PREVISÃO: horizonte futuro
            # ==========================================
            forecast_dates_range = pd.date_range(
                start=forecast_start,
                periods=T,
                freq='D'
            )

            # Observado (verdade)
            ax.plot(forecast_dates_range, obs[idx, :, st_i],
                   label='Observado', color='black', linewidth=2.5)

            # Previsto pelo modelo
            ax.plot(forecast_dates_range, preds[idx, :, st_i],
                   label='Previsto', color='royalblue', linestyle='--', linewidth=2)

            # Baseline de persistência
            if baseline_last is not None:
                base_line = np.full(T, float(baseline_last[idx, st_i]), dtype=float)
                ax.plot(forecast_dates_range, base_line,
                       label='Persistência', color='darkorange',
                       linestyle=':', linewidth=2)

            # Linha vertical marcando início da previsão
            ax.axvline(forecast_start, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.7, label='Início previsão')

            # ==========================================
            # GATE (eixo Y secundário)
            # ==========================================
            if g_seq is not None:
                ax2 = ax.twinx()
                ax2.plot(forecast_dates_range, g_seq[idx, :, st_i],
                        color='seagreen', alpha=0.6, linewidth=1.5,
                        label='Gate y_prev')
                ax2.set_ylim(0, 1.05)
                ax2.set_ylabel('Gate', color='seagreen', fontsize=11)
                ax2.tick_params(axis='y', labelcolor='seagreen')
                twin_axes.append(ax2)

            # ==========================================
            # FORMATAÇÃO
            # ==========================================
            ax.set_title(
                f'Estação {st_id} — Previsão iniciando em {forecast_start.strftime("%Y-%m-%d")}',
                fontsize=12, fontweight='bold'
            )
            ax.set_ylabel('Vazão (m³/s)', fontsize=11)
            ax.grid(True, alpha=0.3)

            # Formatar eixo X com datas
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, (context_days + T) // 10)))

            if st_i == S - 1:
                ax.set_xlabel('Data', fontsize=11)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # ==========================================
        # LEGENDA COMBINADA
        # ==========================================
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l
        for ax2 in twin_axes:
            h2, l2 = ax2.get_legend_handles_labels()
            handles += h2
            labels += l2

        # Remover duplicatas mantendo ordem
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(),
                  loc='upper center', ncol=min(5, len(by_label)),
                  fontsize=10, framealpha=0.95)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()


def plot_metrics_by_horizon(
    metrics: Dict[int, Dict[str, Any]],
    stations: List[int],
    figsize: tuple = (14, 5)
) -> None:
    """
    Plota RMSE e MAE por horizonte para cada estação.

    Args:
        metrics: Dicionário de métricas retornado por compute_flow_metrics
        stations: Lista de IDs das estações
        figsize: Tamanho da figura (width, height)
    """
    n_stations = len(stations)

    fig, axes = plt.subplots(n_stations, 2, figsize=(figsize[0], figsize[1] * n_stations))

    # Garantir que axes seja 2D
    if n_stations == 1:
        axes = axes.reshape(1, -1)

    for st_idx, station in enumerate(stations):
        station_metrics = metrics[station]
        per_horizon = station_metrics["per_horizon"]

        rmse_values = np.array(per_horizon["rmse"])
        mae_values = np.array(per_horizon["mae"])
        horizons = np.arange(1, len(rmse_values) + 1)

        # ==========================================
        # SUBPLOT 1: RMSE
        # ==========================================
        ax_rmse = axes[st_idx, 0]
        ax_rmse.plot(horizons, rmse_values, marker='o', linewidth=2,
                    markersize=5, color='#e74c3c', label='RMSE')
        ax_rmse.fill_between(horizons, 0, rmse_values, alpha=0.2, color='#e74c3c')
        ax_rmse.set_xlabel('Horizonte de Previsão (dias)', fontsize=11)
        ax_rmse.set_ylabel('RMSE (m³/s)', fontsize=11)
        ax_rmse.set_title(f'Estação {station} - RMSE por Horizonte',
                         fontsize=12, fontweight='bold')
        ax_rmse.grid(True, alpha=0.3)
        ax_rmse.legend(loc='best', fontsize=10)

        # ==========================================
        # SUBPLOT 2: MAE
        # ==========================================
        ax_mae = axes[st_idx, 1]
        ax_mae.plot(horizons, mae_values, marker='s', linewidth=2,
                   markersize=5, color='#3498db', label='MAE')
        ax_mae.fill_between(horizons, 0, mae_values, alpha=0.2, color='#3498db')
        ax_mae.set_xlabel('Horizonte de Previsão (dias)', fontsize=11)
        ax_mae.set_ylabel('MAE (m³/s)', fontsize=11)
        ax_mae.set_title(f'Estação {station} - MAE por Horizonte',
                        fontsize=12, fontweight='bold')
        ax_mae.grid(True, alpha=0.3)
        ax_mae.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.show()
