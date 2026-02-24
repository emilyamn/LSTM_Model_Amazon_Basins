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

def plot_full_series_with_d1_forecast(
    preds: np.ndarray,
    obs: np.ndarray,
    stations: List[int],
    forecast_dates: np.ndarray,
    df: pd.DataFrame,
    period_name: str = "Validação",
    baseline_last: Optional[np.ndarray] = None,
    figsize: tuple = (16, 5)
) -> None:
    """
    Plota série temporal completa mostrando apenas previsões D+1 (um dia à frente).
    
    Útil para visualizar o desempenho ao longo de todo o período de validação/teste,
    focando apenas na previsão de curto prazo (1 dia à frente).
    
    Args:
        preds: Previsões, shape (batch, horizon, n_stations)
        obs: Observações, shape (batch, horizon, n_stations)
        stations: Lista de IDs das estações
        forecast_dates: Datas de início de cada previsão, shape (batch,)
        df: DataFrame original com índice temporal
        period_name: Nome do período ('Validação' ou 'Teste')
        baseline_last: Baseline de persistência, shape (batch, n_stations)
        figsize: Tamanho da figura
    """
    B, T, S = preds.shape
    
    # Converter forecast_dates para datetime
    forecast_dates_dt = pd.to_datetime(forecast_dates)
    
    # Determinar período completo
    start_date = forecast_dates_dt.min()
    end_date = forecast_dates_dt.max() + timedelta(days=T-1)
    
    fig, axes = plt.subplots(S, 1, figsize=(figsize[0], figsize[1] * S), sharex=True)
    axes = np.atleast_1d(axes)
    
    for st_i, st_id in enumerate(stations):
        ax = axes[st_i]
        
        # ==========================================
        # SÉRIE OBSERVADA COMPLETA
        # ==========================================
        period_mask = (df.index >= start_date) & (df.index <= end_date)
        period_dates = df.index[period_mask]
        period_obs = df[f"Q_{st_id}"].loc[period_mask].values
        
        ax.plot(period_dates, period_obs,
               label='Observado', color='black', linewidth=1.5, alpha=0.8)
        
        # ==========================================
        # PREVISÕES D+1 (PRIMEIRO DIA DO HORIZONTE)
        # ==========================================
        d1_forecast_dates = []
        d1_forecast_values = []
        
        for idx in range(B):
            # Data da previsão D+1 (primeiro dia após forecast_start)
            d1_date = forecast_dates_dt[idx] + timedelta(days=1)
            d1_value = preds[idx, 0, st_i]  # Primeiro dia do horizonte (índice 0)
            
            d1_forecast_dates.append(d1_date)
            d1_forecast_values.append(d1_value)
        
        ax.scatter(d1_forecast_dates, d1_forecast_values,
                  label='Previsão D+1', color='royalblue', s=20, alpha=0.7, zorder=5)
        
        # ==========================================
        # BASELINE DE PERSISTÊNCIA (OPCIONAL)
        # ==========================================
        if baseline_last is not None:
            baseline_dates = []
            baseline_values = []
            
            for idx in range(B):
                d1_date = forecast_dates_dt[idx] + timedelta(days=1)
                baseline_dates.append(d1_date)
                baseline_values.append(baseline_last[idx, st_i])
            
            ax.scatter(baseline_dates, baseline_values,
                      label='Persistência D+1', color='darkorange', 
                      s=20, alpha=0.5, marker='x', zorder=4)
        
        # ==========================================
        # FORMATAÇÃO
        # ==========================================
        ax.set_title(
            f'Estação {st_id} — Série Completa ({period_name}) com Previsões D+1',
            fontsize=12, fontweight='bold'
        )
        ax.set_ylabel('Vazão (m³/s)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Formatar eixo X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        
        if st_i == S - 1:
            ax.set_xlabel('Data', fontsize=11)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()


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
) -> None:
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
    """
    B, T, S = preds.shape
    n_samples = min(n_samples, B)
    
    # Validar extreme_type
    if extreme_type not in ['max', 'min']:
        raise ValueError("extreme_type deve ser 'max' ou 'min'")
    
    print(f"\n{'='*60}")
    print(f"SELECIONANDO {n_samples} EVENTOS {'MÁXIMOS' if extreme_type == 'max' else 'MÍNIMOS'}")
    print(f"{'='*60}\n")
    
    # Para cada estação, encontrar os índices dos eventos extremos
    for st_i, st_id in enumerate(stations):
        print(f"📊 Estação {st_id}:")
        
        # Calcular valores médios observados para cada previsão
        # (média do horizonte de previsão)
        mean_obs_per_forecast = obs[:, :, st_i].mean(axis=1)
        
        # Selecionar índices dos extremos
        if extreme_type == 'max':
            # Índices dos N maiores valores
            extreme_indices = np.argsort(mean_obs_per_forecast)[-n_samples:][::-1]
            print(f"  Selecionados {n_samples} eventos de MÁXIMA vazão")
        else:
            # Índices dos N menores valores
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
            
            # Definir janela de contexto
            context_start = forecast_start - timedelta(days=context_days)
            
            fig, ax = plt.subplots(1, 1, figsize=(14, 5))
            twin_ax = None
            
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
                twin_ax = ax.twinx()
                twin_ax.plot(forecast_dates_range, g_seq[idx, :, st_i],
                            color='seagreen', alpha=0.6, linewidth=1.5,
                            label='Gate y_prev')
                twin_ax.set_ylim(0, 1.05)
                twin_ax.set_ylabel('Gate', color='seagreen', fontsize=11)
                twin_ax.tick_params(axis='y', labelcolor='seagreen')
            
            # ==========================================
            # FORMATAÇÃO
            # ==========================================
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
            
            # Formatar eixo X
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, (context_days + T) // 10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # ==========================================
            # LEGENDA COMBINADA
            # ==========================================
            handles, labels = ax.get_legend_handles_labels()
            if twin_ax is not None:
                h2, l2 = twin_ax.get_legend_handles_labels()
                handles += h2
                labels += l2
            
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(),
                     loc='best', fontsize=10, framealpha=0.95)
            
            plt.tight_layout()
            plt.show()
    
    print(f"{'='*60}")
    print(f"✅ VISUALIZAÇÃO DE EXTREMOS CONCLUÍDA")
    print(f"{'='*60}\n")