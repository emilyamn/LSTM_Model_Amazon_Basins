"""
Módulo para extração e análise de eventos extremos de vazão.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


@dataclass
class FlowEventThresholds:
    """Limiares para classificação de eventos extremos."""
    station: int
    mean_annual_max: float
    std_annual_max: float
    threshold_extreme_high: float
    threshold_moderate_high: float
    mean_annual_min: float
    std_annual_min: float
    threshold_extreme_low: float
    threshold_moderate_low: float


def compute_annual_thresholds(
    df: pd.DataFrame,
    stations: List[int],
    flow_col_pattern: str = "Q_{}"
) -> Dict[int, FlowEventThresholds]:
    """
    Calcula limiares anuais para eventos extremos.
    """
    print("" + "="*80)
    print("CALCULANDO LIMIARES ANUAIS PARA EVENTOS EXTREMOS")
    print("="*80)

    thresholds = {}

    for station in stations:
        col = flow_col_pattern.format(station)

        if col not in df.columns:
            print(f"⚠️  Coluna {col} não encontrada, pulando estação {station}")
            continue

        df_station = df[[col]].copy()
        df_station['year'] = df_station.index.year

        annual_max = df_station.groupby('year')[col].max()
        annual_min = df_station.groupby('year')[col].min()

        mean_max = float(annual_max.mean())
        std_max = float(annual_max.std())
        mean_min = float(annual_min.mean())
        std_min = float(annual_min.std())

        threshold_extreme_high = mean_max + 1.0 * std_max
        threshold_moderate_high = mean_max
        threshold_extreme_low = mean_min - 1.0 * std_min
        threshold_moderate_low = mean_min

        thresholds[station] = FlowEventThresholds(
            station=station,
            mean_annual_max=mean_max,
            std_annual_max=std_max,
            threshold_extreme_high=threshold_extreme_high,
            threshold_moderate_high=threshold_moderate_high,
            mean_annual_min=mean_min,
            std_annual_min=std_min,
            threshold_extreme_low=threshold_extreme_low,
            threshold_moderate_low=threshold_moderate_low
        )

        print(f"📊 Estação {station}:")
        print(f"  Máximas anuais: média={mean_max:.2f}, std={std_max:.2f}")
        print(f"    Limiar EXTREMO (alta):  Q > {threshold_extreme_high:.2f} m³/s")
        print(f"    Limiar MODERADO (alta): {threshold_moderate_high:.2f} < Q < {threshold_extreme_high:.2f} m³/s")
        print(f"  Mínimas anuais: média={mean_min:.2f}, std={std_min:.2f}")
        print(f"    Limiar EXTREMO (baixa):  Q < {threshold_extreme_low:.2f} m³/s")
        print(f"    Limiar MODERADO (baixa): {threshold_extreme_low:.2f} < Q < {threshold_moderate_low:.2f} m³/s")

    print("" + "="*80 + "")
    return thresholds


def classify_flow_events(
    df: pd.DataFrame,
    thresholds: Dict[int, FlowEventThresholds],
    flow_col_pattern: str = "Q_{}"
) -> Dict[int, pd.Series]:
    """
    Classifica cada dia como: 0=normal, 1=moderado, 2=extremo.
    """
    classifications = {}

    for station, thresh in thresholds.items():
        col = flow_col_pattern.format(station)

        if col not in df.columns:
            continue

        Q = df[col].values
        event_type = np.zeros(len(Q), dtype=int)

        extreme_high = Q > thresh.threshold_extreme_high
        extreme_low = Q < thresh.threshold_extreme_low
        event_type[extreme_high | extreme_low] = 2

        moderate_high = (Q > thresh.threshold_moderate_high) & (Q <= thresh.threshold_extreme_high)
        moderate_low = (Q >= thresh.threshold_extreme_low) & (Q < thresh.threshold_moderate_low)
        event_type[moderate_high | moderate_low] = 1

        classifications[station] = pd.Series(event_type, index=df.index, name=f'event_type_{station}')

    return classifications


def extract_event_window_indices_by_type(
    event_classifications: Dict[int, pd.Series],
    window_dates: np.ndarray,
    stations: List[int],
) -> Dict[str, Dict[int, List[int]]]:
    window_indices = {'extreme': {}, 'moderate': {}, 'normal': {}}
    window_dates_dt = pd.to_datetime(window_dates)

    for station in stations:
        if station not in event_classifications:
            for key in window_indices:
                window_indices[key][station] = []
            continue

        classification = event_classifications[station]
        
        extreme_indices, moderate_indices, normal_indices = [], [], []
        
        for idx, date in enumerate(window_dates_dt):
            if date in classification.index:
                event_val = classification.loc[date]
                
                if event_val == 2:
                    extreme_indices.append(idx)
                elif event_val == 1:
                    moderate_indices.append(idx)
                else:
                    normal_indices.append(idx)
            else:
                # ← NOVA LÓGICA: se data não está no índice, procurar a mais próxima
                nearest_idx = classification.index.get_indexer([date], method='nearest')[0]
                if nearest_idx >= 0:
                    event_val = classification.iloc[nearest_idx]
                    if event_val == 2:
                        extreme_indices.append(idx)
                    elif event_val == 1:
                        moderate_indices.append(idx)
                    else:
                        normal_indices.append(idx)
                else:
                    normal_indices.append(idx)

        window_indices['extreme'][station] = extreme_indices
        window_indices['moderate'][station] = moderate_indices
        window_indices['normal'][station] = normal_indices

    return window_indices

def print_event_statistics(
    event_classifications: Dict[int, pd.Series],
    stations: List[int]
) -> None:
    """Imprime estatísticas sobre eventos extremos."""
    print("="*80)
    print("ESTATÍSTICAS DE EVENTOS EXTREMOS")
    print("="*80)

    for station in stations:
        if station not in event_classifications:
            continue

        classification = event_classifications[station]

        n_total = len(classification)
        n_extreme = (classification == 2).sum()
        n_moderate = (classification == 1).sum()  # ← CORRIGIDO: era n_moderade
        n_normal = (classification == 0).sum()

        pct_extreme = 100.0 * n_extreme / n_total
        pct_moderate = 100.0 * n_moderate / n_total  # ← CORRIGIDO
        pct_normal = 100.0 * n_normal / n_total

        print(f"📊 Estação {station}:")
        print(f"  Total de dias: {n_total}")
        print(f"  Extremos (tipo 2): {n_extreme} dias ({pct_extreme:.2f}%)")
        print(f"  Moderados (tipo 1): {n_moderate} dias ({pct_moderate:.2f}%)")
        print(f"  Normais (tipo 0): {n_normal} dias ({pct_normal:.2f}%)")

    print("="*80 + "")

def plot_events_timeline(
    df: pd.DataFrame,
    event_classifications: Dict[int, pd.Series],
    stations: List[int],
    flow_col_pattern: str = "Q_{}",
    figsize: Tuple[int, int] = (16, 4),
    show: bool = False,
    return_fig: bool = True,
) -> List[plt.Figure]:
    """Plota linha do tempo das vazões com eventos coloridos."""
    n_stations = len(stations)

    figures = []
    colors = {0: 'lightgray', 1: 'orange', 2: 'red'}
    labels = {0: 'Normal', 1: 'Moderado', 2: 'Extremo'}

    for st_idx, station in enumerate(stations):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        if station not in event_classifications or flow_col_pattern.format(station) not in df.columns:
            plt.close(fig)
            continue

        Q = df[flow_col_pattern.format(station)]
        classification = event_classifications[station]

        ax.plot(df.index, Q, color='black', linewidth=0.8, alpha=0.6, zorder=1)

        for event_type in [2, 1, 0]:
            mask = classification == event_type
            dates = classification[mask].index
            values = Q[mask]

            ax.scatter(dates, values,
                      c=colors[event_type],
                      s=3,
                      alpha=0.7,
                      label=labels[event_type],
                      zorder=3 - event_type)

        ax.set_ylabel('Vazão (m³/s)', fontsize=11)
        ax.set_title(f'Estação {station} - Timeline de Eventos', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        
        if show:
            plt.show()
        
        figures.append(fig)
        if not return_fig:
            plt.close(fig)

    return figures if return_fig else None


def analyze_flow_extremes(
    df: pd.DataFrame,
    stations: List[int],
    window_dates: Optional[np.ndarray] = None,  # MUDANÇA: parâmetro renomeado
    flow_col_pattern: str = "Q_{}",
    return_window_indices: bool = True,
) -> Tuple[Dict[int, FlowEventThresholds], Dict[int, pd.Series], Dict[str, Dict[int, List[int]]]]:
    """
    Pipeline completo de análise de eventos extremos.
    
    Args:
        df: DataFrame com índice temporal e colunas de vazão
        stations: Lista de IDs das estações
        window_dates: Array de datas de início de cada janela (opcional) - VEM DO predict_autoregressive
        flow_col_pattern: Padrão do nome da coluna
        return_window_indices: Se True, retorna índices de janelas por tipo

    Returns:
        Tupla com: (thresholds, classifications, window_indices)
    """
    # 1. Calcular limiares
    thresholds = compute_annual_thresholds(df, stations, flow_col_pattern)

    # 2. Classificar dias
    classifications = classify_flow_events(df, thresholds, flow_col_pattern)

    # 3. Estatísticas
    print_event_statistics(classifications, stations)

    # 4. Plot
    plot_events_timeline(df, classifications, stations, flow_col_pattern)

    # 5. Extrair índices de janelas
    window_indices = {}
    if window_dates is not None and return_window_indices:
        window_indices = extract_event_window_indices_by_type(
            classifications, window_dates, stations
        )

        print("" + "="*80)
        print("JANELAS DE PREVISÃO POR TIPO DE EVENTO")
        print("="*80)

        for event_name, indices_dict in window_indices.items():
            print(f"📌 {event_name.upper()}:")
            for station, indices in indices_dict.items():
                print(f"  Estação {station}: {len(indices)} janelas")

        print("" + "="*80 + "")

    return thresholds, classifications, window_indices