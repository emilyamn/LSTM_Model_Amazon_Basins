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
    indices: Optional[List[int]] = None,
    n_samples: Optional[int] = None,
    context_days: int = 30,
    baseline_last: Optional[np.ndarray] = None,
    g_seq: Optional[np.ndarray] = None,
    show: bool = False,
) -> List[plt.Figure]:
    """
    Plota previsões com contexto. Permite selecionar índices específicos ou
    amostras aleatórias. Se ambos forem None, plota todo o batch.
    """
    B, T, S = preds.shape

    # Lógica de seleção de amostras
    if indices is not None:
        sample_indices = indices
    elif n_samples is not None:
        n_samples = min(n_samples, B)
        sample_indices = np.random.choice(B, n_samples, replace=False)
    else:
        sample_indices = np.arange(B)

    figures = []

    for idx in sample_indices:
        forecast_start = pd.to_datetime(forecast_dates[idx])
        context_start = forecast_start - timedelta(days=context_days)

        fig, axes = plt.subplots(S, 1, figsize=(14, 4 * S), sharex=True)
        axes = np.atleast_1d(axes)

        for st_i, st_id in enumerate(stations):
            ax = axes[st_i]
            # Contexto histórico
            context_mask = (df.index >= context_start) & (df.index < forecast_start)
            ax.plot(df.index[context_mask], df[f"Q_{st_id}"].loc[context_mask].values,
                   color='gray', linewidth=1.5, alpha=0.6, label='Histórico')

            # Previsão e Observado
            forecast_dates_range = pd.date_range(start=forecast_start, periods=T, freq='D')
            ax.plot(forecast_dates_range, obs[idx, :, st_i], label='Observado', color='black', linewidth=2.5)
            ax.plot(forecast_dates_range, preds[idx, :, st_i], label='Previsto', color='royalblue', linestyle='--', linewidth=2)

            if baseline_last is not None:
                ax.plot(forecast_dates_range, np.full(T, baseline_last[idx, st_i]),
                        label='Persistência', color='darkorange', linestyle=':')

            ax.axvline(forecast_start, color='red', linestyle='--', alpha=0.7)
            ax.set_title(f"Estação {st_id} - Amostra Index: {idx}")

            if g_seq is not None:
                ax2 = ax.twinx()
                ax2.plot(forecast_dates_range, g_seq[idx, :, st_i], color='seagreen', alpha=0.4, label='Gate')
                ax2.set_ylim(0, 1.1)

        plt.tight_layout()
        if show: plt.show()
        figures.append(fig)

    return figures

def plot_metrics_by_horizon(
    metrics: Dict[int, Dict[str, Any]],
    stations: List[int],
    figsize: tuple = (14, 5),
    show: bool = False,
) -> plt.Figure:
    n_stations = len(stations)
    fig, axes = plt.subplots(n_stations, 2, figsize=(figsize[0], figsize[1] * n_stations))
    if n_stations == 1: axes = axes.reshape(1, -1)

    for st_idx, station in enumerate(stations):
        m = metrics[station]["per_horizon"]
        h = np.arange(1, len(m["rmse"]) + 1)
        axes[st_idx, 0].plot(h, m["rmse"], marker='o', color='#e74c3c', label='RMSE')
        axes[st_idx, 1].plot(h, m["mae"], marker='s', color='#3498db', label='MAE')
        axes[st_idx, 0].set_title(f"Estação {station} - RMSE por Horizonte")
        axes[st_idx, 1].set_title(f"Estação {station} - MAE por Horizonte")
        for ax in axes[st_idx]: ax.grid(True, alpha=0.3); ax.legend()

    plt.tight_layout()
    if show: plt.show()
    return fig

def plot_full_series_with_d1_forecast(
    preds: np.ndarray, obs: np.ndarray, stations: List[int],
    forecast_dates: np.ndarray, df: pd.DataFrame,
    show: bool = False, figsize: tuple = (16, 5),
) -> plt.Figure:
    S = preds.shape[2]
    fig, axes = plt.subplots(S, 1, figsize=(figsize[0], figsize[1] * S), sharex=True)
    axes = np.atleast_1d(axes)

    dates_dt = pd.to_datetime(forecast_dates)
    for st_i, st_id in enumerate(stations):
        ax = axes[st_i]
        ax.plot(df.index, df[f"Q_{st_id}"], color='black', alpha=0.2, label='Obs. Total')
        ax.scatter(dates_dt, preds[:, 0, st_i], color='royalblue', s=10, label='D+1 Previsto', alpha=0.7)
        ax.set_title(f"Série Temporal Completa vs Previsão D+1 - Estação {st_id}")
        ax.legend()

    plt.tight_layout()
    if show: plt.show()
    return fig

def plot_predictions_extremes(
    preds: np.ndarray,
    obs: np.ndarray,
    stations: List[int],
    df: pd.DataFrame,
    forecast_dates: np.ndarray,
    n_samples: int = 3,
    extreme_type: str = 'max',
    context_days: int = 30,
    baseline_last: Optional[np.ndarray] = None,
    g_seq: Optional[np.ndarray] = None,
    show: bool = False,
    return_fig: bool = True,
) -> List[plt.Figure]:
    """
    Plota previsões para eventos extremos (máximas ou mínimas).

    Similar a plot_predictions_with_context, mas seleciona automaticamente
    os N eventos de maior ou menor vazão observada para análise detalhada.

    Args:
        preds: Previsões, shape (batch, horizon, n_stations)
        obs: Observações, shape (batch, horizon, n_stations)
        stations: Lista de IDs das estações
        df: DataFrame original com índice temporal
        forecast_dates: Datas de início de cada previsão, shape (batch,)
        n_samples: Número de eventos extremos para plotar
        extreme_type: 'max' para máximas, 'min' para mínimas
        context_days: Quantos dias anteriores mostrar como contexto
        baseline_last: Baseline de persistência, shape (batch, n_stations)
        g_seq: Sequência de gates, shape (batch, horizon, n_stations)
        show: Se True, exibe os gráficos
        return_fig: Se True, retorna as figuras

    Returns:
        Lista de figuras (plt.Figure) se return_fig=True, caso contrário None
    """

    B, T, S = preds.shape
    n_samples = min(n_samples, B)

    # Validar extreme_type
    if extreme_type not in ['max', 'min']:
        raise ValueError("extreme_type deve ser 'max' ou 'min'")

    print(f"{'='*60}")
    print(f"SELECIONANDO {n_samples} EVENTOS {'MÁXIMOS' if extreme_type == 'max' else 'MÍNIMOS'}")
    print(f"{'='*60}")

    figures = []

    # Para cada estação, encontrar os índices dos eventos extremos
    for st_i, st_id in enumerate(stations):
        print(f"📊 Estação {st_id}:")

        # Calcular valores médios observados para cada previsão
        mean_obs_per_forecast = obs[:, :, st_i].mean(axis=1)

        # Selecionar índices dos extremos
        if extreme_type == 'max':
            extreme_indices = np.argsort(mean_obs_per_forecast)[-n_samples:][::-1]
            print(f"  Selecionados {n_samples} eventos de MÁXIMA vazão")
        else:
            extreme_indices = np.argsort(mean_obs_per_forecast)[:n_samples]
            print(f"  Selecionados {n_samples} eventos de MÍNIMA vazão")

        # Mostrar valores selecionados
        for rank, idx in enumerate(extreme_indices, 1):
            date = pd.to_datetime(forecast_dates[idx]).strftime('%Y-%m-%d')
            value = mean_obs_per_forecast[idx]
            print(f"    {rank}º: {date} — Vazão média: {value:.2f} m³/s")

        print()

        # Plotar cada evento extremo
        for idx in extreme_indices:
            forecast_start = pd.to_datetime(forecast_dates[idx])
            context_start = forecast_start - timedelta(days=context_days)

            fig, ax = plt.subplots(1, 1, figsize=(14, 5))
            twin_ax = None

            # CONTEXTO: dados históricos
            context_mask = (df.index >= context_start) & (df.index < forecast_start)
            context_dates = df.index[context_mask]
            context_values = df[f"Q_{st_id}"].loc[context_mask].values

            if len(context_dates) > 0:
                ax.plot(context_dates, context_values,
                       color='gray', linewidth=1.5, alpha=0.6,
                       label='Histórico (contexto)')

            # PREVISÃO: horizonte futuro
            forecast_dates_range = pd.date_range(
                start=forecast_start,
                periods=T,
                freq='D'
            )

            ax.plot(forecast_dates_range, obs[idx, :, st_i],
                   label='Observado', color='black', linewidth=2.5)
            ax.plot(forecast_dates_range, preds[idx, :, st_i],
                   label='Previsto', color='royalblue', linestyle='--', linewidth=2)

            if baseline_last is not None:
                base_line = np.full(T, float(baseline_last[idx, st_i]), dtype=float)
                ax.plot(forecast_dates_range, base_line,
                       label='Persistência', color='darkorange',
                       linestyle=':', linewidth=2)

            ax.axvline(forecast_start, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.7, label='Início previsão')

            # GATE (eixo Y secundário)
            if g_seq is not None:
                twin_ax = ax.twinx()
                twin_ax.plot(forecast_dates_range, g_seq[idx, :, st_i],
                            color='seagreen', alpha=0.6, linewidth=1.5,
                            label='Gate y_prev')
                twin_ax.set_ylim(0, 1.05)
                twin_ax.set_ylabel('Gate', color='seagreen', fontsize=11)
                twin_ax.tick_params(axis='y', labelcolor='seagreen')

            # FORMATAÇÃO
            event_type_str = "MÁXIMA" if extreme_type == 'max' else "MÍNIMA"
            mean_q = mean_obs_per_forecast[idx]

            ax.set_title(
                f'Estação {st_id} — Evento de {event_type_str} '
                f'({forecast_start.strftime("%Y-%m-%d")}, Q_média={mean_q:.2f} m³/s)',
                fontsize=12, fontweight='bold'
            )
            ax.set_ylabel('Vazão (m³/s)', fontsize=11)
            ax.set_xlabel('Data', fontsize=11)
            ax.grid(True, alpha=0.3)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, (context_days + T) // 10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            handles, labels = ax.get_legend_handles_labels()
            if twin_ax is not None:
                h2, l2 = twin_ax.get_legend_handles_labels()
                handles += h2
                labels += l2

            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(),
                     loc='best', fontsize=10, framealpha=0.95)

            plt.tight_layout()

            if show:
                plt.show()

            figures.append(fig)
            if not return_fig:
                plt.close(fig)

    print(f"{'='*60}")
    print("✅ VISUALIZAÇÃO DE EXTREMOS CONCLUÍDA")
    print(f"{'='*60}")

    return figures if return_fig else None


def plot_metrics_by_horizon_comparison(
    metrics_by_type: Dict[str, Dict[int, Dict[str, Any]]],
    stations: List[int],
    figsize: tuple = (14, 5),
    show: bool = False,
    return_fig: bool = True,
    metrics_to_plot: List[str] = None,
) -> plt.Figure:
    """
    Plota métricas por horizonte comparando diferentes tipos de eventos.

    Args:
        metrics_by_type: Dicionário {event_type: {station: metrics}}
        stations: Lista de IDs das estações
        figsize: Tamanho da figura
        show: Se True, exibe o gráfico
        return_fig: Se True, retorna a figura
        metrics_to_plot: Lista de métricas a plotar (padrão: ['rmse', 'mae'])

    Returns:
        Figura matplotlib
    """

    if metrics_to_plot is None:
        metrics_to_plot = ['rmse', 'mae']

    n_stations = len(stations)
    event_types = list(metrics_by_type.keys())
    colors = {'extreme': '#e74c3c', 'moderate': '#f39c12', 'normal': '#95a5a6',
              'extreme_high': '#c0392b', 'extreme_low': '#8e44ad'}
    linestyles = {'extreme': '-', 'moderate': '--', 'normal': ':',
                  'extreme_high': '-', 'extreme_low': '--'}

    # Em vez de plt.cm.tab10, use:
    tab10 = plt.get_cmap('tab10')

    # Para obter as cores:
    for et in event_types:
        if et not in colors:
            colors[et] = tab10(hash(et) % 10)
            linestyles[et] = '-'

    fig, axes = plt.subplots(n_stations, len(metrics_to_plot),
                            figsize=(figsize[0] * len(metrics_to_plot), figsize[1] * n_stations))
    if n_stations == 1 and len(metrics_to_plot) == 1:
        axes = axes.reshape(1, 1)
    elif n_stations == 1:
        axes = axes.reshape(1, -1)
    elif len(metrics_to_plot) == 1:
        axes = axes.reshape(-1, 1)

    for st_idx, station in enumerate(stations):
        for met_idx, metric_name in enumerate(metrics_to_plot):
            if len(metrics_to_plot) == 1:
                ax = axes[st_idx]
            else:
                ax = axes[st_idx, met_idx]

            for event_type in event_types:
                if event_type not in metrics_by_type:
                    continue
                if station not in metrics_by_type[event_type]:
                    continue

                per_horizon = metrics_by_type[event_type][station]['per_horizon']

                if metric_name in per_horizon:
                    metric_values = np.array(per_horizon[metric_name])
                    horizons = np.arange(1, len(metric_values) + 1)

                    ax.plot(horizons, metric_values,
                           marker='o',
                           linewidth=2,
                           markersize=5,
                           color=colors.get(event_type, '#333333'),
                           linestyle=linestyles.get(event_type, '-'),
                           label=event_type.capitalize())

            ax.set_xlabel('Horizonte de Previsão (dias)', fontsize=11)
            ax.set_ylabel(f'{metric_name.upper()}', fontsize=11)
            ax.set_title(f'Estação {station} - {metric_name.upper()} por Horizonte',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    if show: plt.show()
    if not return_fig:
        plt.close(fig)
    return fig
