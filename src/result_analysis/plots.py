"""
Módulo para visualização de resultados do modelo.
"""

from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ===========================================================================
# HELPER COMPARTILHADO
# ===========================================================================

def _resolve_save_dir(
    subdir: str = "plots",
    mode: str = "test",
    path: Optional[str] = None,
) -> Path:
    """
    Resolve o diretório de salvamento.

    Prioridade:
      1. ``path`` fornecido explicitamente.
      2. Último experimento em ``outputs/experiments/`` →
         ``predictions_{mode}/{subdir}/``.
      3. Fallback: ``outputs/{subdir}/``.

    Args:
        subdir: Subpasta dentro do modo ("plots" ou "raw").
        mode:   "test" ou "operational".
        path:   Caminho absoluto customizado (ignora mode/subdir).

    Returns:
        Path do diretório (já criado).
    """
    if path is not None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    from src.utils.experiment_utils import get_experiments_base_dir
    experiments_root = get_experiments_base_dir()

    if experiments_root.exists():
        experiment_dirs = sorted(
            [d for d in experiments_root.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if experiment_dirs:
            target = experiment_dirs[0] / f"predictions_{mode}" / subdir
            target.mkdir(parents=True, exist_ok=True)
            return target

    fallback = Path("outputs") / subdir
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _save_figure(
    fig: plt.Figure,
    filename: str,
    save: bool = True,
    mode: str = "test",
    path: Optional[str] = None,
    dpi: int = 150,
) -> Optional[Path]:
    """Salva uma figura no diretório resolvido."""
    if not save:
        return None

    output_dir = _resolve_save_dir(subdir="plots", mode=mode, path=path)
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"💾 Salvo: {filepath}")
    return filepath


# ===========================================================================
# FUNÇÕES DE PLOT
# ===========================================================================

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
    save: bool = True,
    mode: str = "test",
    path: Optional[str] = None,
    dpi: int = 150,
) -> List[plt.Figure]:
    """
    Plota previsões com contexto histórico.

    Args:
        preds:         Previsões, shape (batch, horizon, n_stations).
        obs:           Observações, shape (batch, horizon, n_stations).
        stations:      Lista de IDs das estações.
        df:            DataFrame original com índice temporal.
        forecast_dates: Datas de início de cada previsão, shape (batch,).
        indices:       Índices específicos a plotar.
        n_samples:     Número de amostras aleatórias (ignorado se indices fornecido).
        context_days:  Dias de contexto histórico antes da previsão.
        baseline_last: Baseline de persistência, shape (batch, n_stations).
        g_seq:         Sequência de gates, shape (batch, horizon, n_stations).
        show:          Se True, exibe as figuras.
        save:          Se True, salva as figuras.
        mode:          "test" ou "operational".
        path:          Caminho customizado. Se None, usa o último experimento.
        dpi:           Resolução das imagens salvas.

    Returns:
        Lista de figuras matplotlib.
    """
    B, T, S = preds.shape

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

            context_mask = (df.index >= context_start) & (df.index < forecast_start)
            ax.plot(
                df.index[context_mask],
                df[f"Q_{st_id}"].loc[context_mask].values,
                color="gray", linewidth=1.5, alpha=0.6, label="Histórico",
            )

            forecast_dates_range = pd.date_range(start=forecast_start, periods=T, freq="D")

            # No modo operational, observado só até reference_date (forecast_start)
            if mode == "operational":
                # Não plotar observado no horizonte — não há dados
                pass
            else:
                ax.plot(forecast_dates_range, obs[idx, :, st_i],
                        label="Observado", color="black", linewidth=2.5)

            ax.plot(forecast_dates_range, preds[idx, :, st_i],
                    label="Previsto", color="royalblue", linestyle="--", linewidth=2)

            if baseline_last is not None:
                ax.plot(
                    forecast_dates_range,
                    np.full(T, baseline_last[idx, st_i]),
                    label="Persistência", color="darkorange", linestyle=":",
                )

            ax.axvline(forecast_start, color="red", linestyle="--", alpha=0.7)
            ax.set_title(f"Estação {st_id} — Amostra Index: {idx}")
            ax.legend(loc="best", fontsize=10)
            ax.grid(True, alpha=0.3)

            if g_seq is not None:
                ax2 = ax.twinx()
                ax2.plot(forecast_dates_range, g_seq[idx, :, st_i],
                         color="seagreen", alpha=0.4, label="Gate")
                ax2.set_ylim(0, 1.1)

        plt.tight_layout()

        filename = f"context_idx{idx}_st{'_'.join(map(str, stations))}.png"
        _save_figure(fig, filename, save=save, mode=mode, path=path, dpi=dpi)

        if show:
            plt.show()

        figures.append(fig)

    return figures


def plot_metrics_by_horizon(
    metrics: Dict[int, Dict[str, Any]],
    stations: List[int],
    figsize: tuple = (14, 5),
    show: bool = False,
    save: bool = True,
    mode: str = "test",
    path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plota RMSE e MAE por horizonte de previsão para cada estação.

    Args:
        metrics:   Dicionário {station: metrics} retornado por compute_flow_metrics.
        stations:  Lista de IDs das estações.
        figsize:   Tamanho base da figura (largura, altura por estação).
        show:      Se True, exibe a figura.
        save:      Se True, salva a figura.
        mode:      "test" ou "operational".
        path:      Caminho customizado. Se None, usa o último experimento.
        dpi:       Resolução da imagem salva.

    Returns:
        Figura matplotlib.
    """
    n_stations = len(stations)
    fig, axes = plt.subplots(n_stations, 2, figsize=(figsize[0], figsize[1] * n_stations))
    if n_stations == 1:
        axes = axes.reshape(1, -1)

    for st_idx, station in enumerate(stations):
        m = metrics[station]["per_horizon"]
        h = np.arange(1, len(m["rmse"]) + 1)

        axes[st_idx, 0].plot(h, m["rmse"], marker="o", color="#e74c3c", label="RMSE")
        axes[st_idx, 1].plot(h, m["mae"], marker="s", color="#3498db", label="MAE")
        axes[st_idx, 0].set_title(f"Estação {station} — RMSE por Horizonte")
        axes[st_idx, 1].set_title(f"Estação {station} — MAE por Horizonte")

        for ax in axes[st_idx]:
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.tight_layout()

    _save_figure(fig, "metrics_by_horizon.png", save=save, mode=mode, path=path, dpi=dpi)

    if show:
        plt.show()

    return fig


def plot_full_series_with_d1_forecast(
    preds: np.ndarray,
    obs: np.ndarray,
    stations: List[int],
    forecast_dates: np.ndarray,
    df: pd.DataFrame,
    period_name: str = "Validação",
    baseline_last: Optional[np.ndarray] = None,
    figsize: tuple = (16, 5),
    show: bool = False,
    save: bool = True,
    mode: str = "test",
    path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plota série temporal completa com previsões D+1 (um dia à frente).

    Args:
        preds:         Previsões, shape (batch, horizon, n_stations).
        obs:           Observações, shape (batch, horizon, n_stations).
        stations:      Lista de IDs das estações.
        forecast_dates: Datas de início de cada previsão, shape (batch,).
        df:            DataFrame original com índice temporal.
        period_name:   Nome do período ('Validação' ou 'Teste').
        baseline_last: Baseline de persistência, shape (batch, n_stations).
        figsize:       Tamanho base da figura.
        show:          Se True, exibe a figura.
        save:          Se True, salva a figura.
        mode:          "test" ou "operational".
        path:          Caminho customizado. Se None, usa o último experimento.
        dpi:           Resolução da imagem salva.

    Returns:
        Figura matplotlib.
    """
    B, T, S = preds.shape
    forecast_dates_dt = pd.to_datetime(forecast_dates)

    start_date = forecast_dates_dt.min()
    end_date = forecast_dates_dt.max() + timedelta(days=T - 1)

    fig, axes = plt.subplots(S, 1, figsize=(figsize[0], figsize[1] * S), sharex=True)
    axes = np.atleast_1d(axes)

    for st_i, st_id in enumerate(stations):
        ax = axes[st_i]

        period_mask = (df.index >= start_date) & (df.index <= end_date)
        ax.plot(
            df.index[period_mask],
            df[f"Q_{st_id}"].loc[period_mask].values,
            label="Observado", color="black", linewidth=1.5, alpha=0.8,
        )

        d1_dates = [forecast_dates_dt[i] for i in range(B)]
        d1_values = [preds[i, 0, st_i] for i in range(B)]
        ax.scatter(d1_dates, d1_values,
                   label="Previsão D+1", color="royalblue", s=20, alpha=0.7, zorder=5)

        if baseline_last is not None:
            ax.scatter(
                d1_dates,
                [baseline_last[i, st_i] for i in range(B)],
                label="Persistência D+1", color="darkorange",
                s=20, alpha=0.5, marker="x", zorder=4,
            )

        ax.set_title(
            f"Estação {st_id} — Série Completa ({period_name}) com Previsões D+1",
            fontsize=12, fontweight="bold",
        )
        ax.set_ylabel("Vazão (m³/s)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

        if st_i == S - 1:
            ax.set_xlabel("Data", fontsize=11)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    period_slug = period_name.lower().replace(" ", "_")
    _save_figure(fig, f"full_series_d1_{period_slug}.png",
                 save=save, mode=mode, path=path, dpi=dpi)

    if show:
        plt.show()

    return fig


def plot_predictions_extremes(
    preds: np.ndarray,
    obs: np.ndarray,
    stations: List[int],
    df: pd.DataFrame,
    forecast_dates: np.ndarray,
    n_samples: int = 3,
    extreme_type: str = "max",
    context_days: int = 30,
    baseline_last: Optional[np.ndarray] = None,
    g_seq: Optional[np.ndarray] = None,
    show: bool = False,
    save: bool = True,
    mode: str = "test",
    path: Optional[str] = None,
    dpi: int = 150,
) -> List[plt.Figure]:
    """
    Plota previsões para eventos extremos (máximas ou mínimas).

    Args:
        preds:         Previsões, shape (batch, horizon, n_stations).
        obs:           Observações, shape (batch, horizon, n_stations).
        stations:      Lista de IDs das estações.
        df:            DataFrame original com índice temporal.
        forecast_dates: Datas de início de cada previsão, shape (batch,).
        n_samples:     Número de eventos extremos a plotar.
        extreme_type:  'max' para máximas, 'min' para mínimas.
        context_days:  Dias de contexto histórico antes da previsão.
        baseline_last: Baseline de persistência, shape (batch, n_stations).
        g_seq:         Sequência de gates, shape (batch, horizon, n_stations).
        show:          Se True, exibe as figuras.
        save:          Se True, salva as figuras.
        mode:          "test" ou "operational".
        path:          Caminho customizado. Se None, usa o último experimento.
        dpi:           Resolução das imagens salvas.

    Returns:
        Lista de figuras matplotlib.
    """
    B, T, S = preds.shape
    n_samples = min(n_samples, B)

    if extreme_type not in ("max", "min"):
        raise ValueError("extreme_type deve ser 'max' ou 'min'")

    event_label = "MÁXIMOS" if extreme_type == "max" else "MÍNIMOS"
    print(f"{'='*60}")
    print(f"SELECIONANDO {n_samples} EVENTOS {event_label}")
    print(f"{'='*60}")

    figures = []

    for st_i, st_id in enumerate(stations):
        print(f"📊 Estação {st_id}:")
        mean_obs = obs[:, :, st_i].mean(axis=1)

        if extreme_type == "max":
            extreme_indices = np.argsort(mean_obs)[-n_samples:][::-1]
        else:
            extreme_indices = np.argsort(mean_obs)[:n_samples]

        for rank, idx in enumerate(extreme_indices, 1):
            date = pd.to_datetime(forecast_dates[idx]).strftime("%Y-%m-%d")
            print(f"  {rank}º: {date} — Vazão média: {mean_obs[idx]:.2f} m³/s")

        print()

        for idx in extreme_indices:
            forecast_start = pd.to_datetime(forecast_dates[idx])
            context_start = forecast_start - timedelta(days=context_days)
            forecast_dates_range = pd.date_range(start=forecast_start, periods=T, freq="D")

            fig, ax = plt.subplots(1, 1, figsize=(14, 5))
            twin_ax = None

            context_mask = (df.index >= context_start) & (df.index < forecast_start)
            if context_mask.any():
                ax.plot(
                    df.index[context_mask],
                    df[f"Q_{st_id}"].loc[context_mask].values,
                    color="gray", linewidth=1.5, alpha=0.6, label="Histórico (contexto)",
                )

            ax.plot(forecast_dates_range, obs[idx, :, st_i],
                    label="Observado", color="black", linewidth=2.5)
            ax.plot(forecast_dates_range, preds[idx, :, st_i],
                    label="Previsto", color="royalblue", linestyle="--", linewidth=2)

            if baseline_last is not None:
                ax.plot(
                    forecast_dates_range,
                    np.full(T, float(baseline_last[idx, st_i])),
                    label="Persistência", color="darkorange", linestyle=":", linewidth=2,
                )

            ax.axvline(forecast_start, color="red", linestyle="--",
                       linewidth=1.5, alpha=0.7, label="Início previsão")

            if g_seq is not None:
                twin_ax = ax.twinx()
                twin_ax.plot(forecast_dates_range, g_seq[idx, :, st_i],
                             color="seagreen", alpha=0.6, linewidth=1.5, label="Gate y_prev")
                twin_ax.set_ylim(0, 1.05)
                twin_ax.set_ylabel("Gate", color="seagreen", fontsize=11)
                twin_ax.tick_params(axis="y", labelcolor="seagreen")

            ax.set_title(
                f"Estação {st_id} — Evento de {event_label} "
                f"({forecast_start.strftime('%Y-%m-%d')}, Q_média={mean_obs[idx]:.2f} m³/s)",
                fontsize=12, fontweight="bold",
            )
            ax.set_ylabel("Vazão (m³/s)", fontsize=11)
            ax.set_xlabel("Data", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(
                mdates.DayLocator(interval=max(1, (context_days + T) // 10))
            )
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

            handles, labels = ax.get_legend_handles_labels()
            if twin_ax is not None:
                h2, l2 = twin_ax.get_legend_handles_labels()
                handles += h2
                labels += l2
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=10)

            plt.tight_layout()

            date_slug = forecast_start.strftime("%Y%m%d")
            filename = f"extreme_{extreme_type}_st{st_id}_{date_slug}.png"
            _save_figure(fig, filename, save=save, mode=mode, path=path, dpi=dpi)

            if show:
                plt.show()

            figures.append(fig)

    print(f"{'='*60}")
    print("✅ VISUALIZAÇÃO DE EXTREMOS CONCLUÍDA")
    print(f"{'='*60}")

    return figures


def plot_metrics_by_horizon_comparison(
    metrics_by_type: Dict[str, Dict[int, Dict[str, Any]]],
    stations: List[int],
    figsize: tuple = (14, 5),
    metrics_to_plot: Optional[List[str]] = None,
    show: bool = False,
    save: bool = True,
    mode: str = "test",
    path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plota métricas por horizonte comparando diferentes tipos de eventos.

    Args:
        metrics_by_type: Dicionário {event_type: {station: metrics}}.
        stations:        Lista de IDs das estações.
        figsize:         Tamanho base da figura.
        metrics_to_plot: Métricas a plotar (padrão: ['rmse', 'mae']).
        show:            Se True, exibe a figura.
        save:            Se True, salva a figura.
        mode:            "test" ou "operational".
        path:            Caminho customizado. Se None, usa o último experimento.
        dpi:             Resolução da imagem salva.

    Returns:
        Figura matplotlib.
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["rmse", "mae"]

    event_types = list(metrics_by_type.keys())
    n_stations = len(stations)

    COLORS = {
        "extreme": "#e74c3c", "moderate": "#f39c12", "normal": "#95a5a6",
        "extreme_high": "#c0392b", "extreme_low": "#8e44ad",
    }
    LINESTYLES = {
        "extreme": "-", "moderate": "--", "normal": ":",
        "extreme_high": "-", "extreme_low": "--",
    }
    tab10 = plt.get_cmap("tab10")
    for et in event_types:
        if et not in COLORS:
            COLORS[et] = tab10(hash(et) % 10)
            LINESTYLES[et] = "-"

    fig, axes = plt.subplots(
        n_stations, len(metrics_to_plot),
        figsize=(figsize[0] * len(metrics_to_plot), figsize[1] * n_stations),
    )
    # Normalizar para sempre ser 2D
    axes = np.atleast_2d(axes)
    if n_stations == 1 and len(metrics_to_plot) > 1:
        axes = axes.reshape(1, -1)
    elif n_stations > 1 and len(metrics_to_plot) == 1:
        axes = axes.reshape(-1, 1)

    for st_idx, station in enumerate(stations):
        for met_idx, metric_name in enumerate(metrics_to_plot):
            ax = axes[st_idx, met_idx]

            for event_type in event_types:
                station_data = metrics_by_type.get(event_type, {}).get(station)
                if station_data is None:
                    continue
                per_horizon = station_data.get("per_horizon", {})
                if metric_name not in per_horizon:
                    continue

                values = np.array(per_horizon[metric_name])
                horizons = np.arange(1, len(values) + 1)
                ax.plot(
                    horizons, values,
                    marker="o", linewidth=2, markersize=5,
                    color=COLORS.get(event_type, "#333333"),
                    linestyle=LINESTYLES.get(event_type, "-"),
                    label=event_type.capitalize(),
                )

            ax.set_xlabel("Horizonte de Previsão (dias)", fontsize=11)
            ax.set_ylabel(metric_name.upper(), fontsize=11)
            ax.set_title(f"Estação {station} — {metric_name.upper()} por Horizonte",
                         fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=10)

    plt.tight_layout()

    metrics_slug = "_".join(metrics_to_plot)
    _save_figure(fig, f"metrics_comparison_{metrics_slug}.png",
                 save=save, mode=mode, path=path, dpi=dpi)

    if show:
        plt.show()

    return fig


def plot_forecast_horizons_analysis(
    preds: np.ndarray,
    obs: np.ndarray,
    stations: List[int],
    forecast_dates: np.ndarray,
    df: pd.DataFrame,
    figsize: tuple = (16, 5),
    show: bool = False,
    save: bool = True,
    mode: str = "test",
    path: Optional[str] = None,
    dpi: int = 150,
) -> List[plt.Figure]:
    """
    Análise completa de horizontes de previsão.

    Gera T+1 figuras:
      - Gráfico principal: Observado + D+1 + D+horizonte_máximo
      - Um gráfico por horizonte: Observado + D+x (D+1 até D+T)

    Args:
        preds:         Previsões, shape (batch, horizon, n_stations).
        obs:           Observações, shape (batch, horizon, n_stations).
        stations:      Lista de IDs das estações.
        forecast_dates: Datas de início de cada previsão, shape (batch,).
        df:            DataFrame original com índice temporal.
        figsize:       Tamanho base da figura.
        show:          Se True, exibe as figuras.
        save:          Se True, salva as figuras.
        mode:          "test" ou "operational".
        path:          Caminho customizado. Se None, usa o último experimento.
        dpi:           Resolução das imagens salvas.

    Returns:
        Lista de figuras matplotlib (principal + uma por horizonte).
    """
    MODE_NAMES = {"test": "Teste", "operational": "Operacional"}
    mode_display = MODE_NAMES.get(mode, mode.capitalize())

    B, T, S = preds.shape
    forecast_dates_dt = pd.to_datetime(forecast_dates)
    start_date = forecast_dates_dt.min()
    end_date = forecast_dates_dt.max() + timedelta(days=T - 1)

    figures = []

    # ------------------------------------------------------------------
    # Gráfico principal: D+1 e D+T
    # ------------------------------------------------------------------
    print(f"\n📊 Criando gráfico principal ({mode_display}: D+1 e D+{T})...")

    fig_main, axes = plt.subplots(S, 1, figsize=(figsize[0], figsize[1] * S), sharex=True)
    axes = np.atleast_1d(axes)

    for st_i, st_id in enumerate(stations):
        ax = axes[st_i]

        period_mask = (df.index >= start_date) & (df.index <= end_date)
        ax.plot(df.index[period_mask], df[f"Q_{st_id}"].loc[period_mask].values,
                label="Observado", color="black", linewidth=1.5, alpha=0.8)

        d1_dates = [forecast_dates_dt[i] for i in range(B)]
        ax.scatter(d1_dates, [preds[i, 0, st_i] for i in range(B)],
                   label="Previsão D+1", color="royalblue", s=20, alpha=0.7, zorder=5)

        dmax_dates = [forecast_dates_dt[i] + timedelta(days=T - 1) for i in range(B)]
        ax.scatter(dmax_dates, [preds[i, T - 1, st_i] for i in range(B)],
                   label=f"Previsão D+{T}", color="orange", s=20, alpha=0.7,
                   marker="s", zorder=5)

        ax.set_title(f"Estação {st_id} — Série {mode_display} Completa (D+1 e D+{T})",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Vazão (m³/s)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

        if st_i == S - 1:
            ax.set_xlabel("Data", fontsize=11)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    _save_figure(fig_main, f"{mode}_d1_dmax.png", save=save, mode=mode, path=path, dpi=dpi)
    if show:
        plt.show()
    figures.append(fig_main)

    # ------------------------------------------------------------------
    # Um gráfico por horizonte: D+1 até D+T
    # ------------------------------------------------------------------
    print(f"\n📊 Criando {T} gráficos individuais (D+1 até D+{T})...")

    for horizon in range(T):
        horizon_day = horizon + 1
        print(f"  → Criando gráfico D+{horizon_day}...")

        fig_h, axes_h = plt.subplots(S, 1, figsize=(figsize[0], figsize[1] * S), sharex=True)
        axes_h = np.atleast_1d(axes_h)

        for st_i, st_id in enumerate(stations):
            ax = axes_h[st_i]

            period_mask = (df.index >= start_date) & (df.index <= end_date)
            ax.plot(df.index[period_mask], df[f"Q_{st_id}"].loc[period_mask].values,
                    label="Observado", color="black", linewidth=1.5, alpha=0.8)

            dx_dates = [forecast_dates_dt[i] + timedelta(days=horizon) for i in range(B)]
            ax.scatter(dx_dates, [preds[i, horizon, st_i] for i in range(B)],
                       label=f"Previsão D+{horizon_day}", color="green",
                       s=20, alpha=0.7, zorder=5)

            ax.set_title(
                f"Estação {st_id} — Série {mode_display} com Previsões D+{horizon_day}",
                fontsize=12, fontweight="bold",
            )
            ax.set_ylabel("Vazão (m³/s)", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=10)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())

            if st_i == S - 1:
                ax.set_xlabel("Data", fontsize=11)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        _save_figure(fig_h, f"{mode}_d{horizon_day}.png",
                     save=save, mode=mode, path=path, dpi=dpi)
        if show:
            plt.show()
        figures.append(fig_h)

    print(f"\n✅ Total de {len(figures)} gráficos criados!")
    return figures
