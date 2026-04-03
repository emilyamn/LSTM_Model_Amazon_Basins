"""
Módulo para cálculo de métricas de desempenho do modelo.
"""

from typing import Dict, Any, Optional, Sequence, List, Union
import numpy as np
import pandas as pd
import json
from pathlib import Path

    
def compute_flow_metrics(
    preds: np.ndarray,
    obs: np.ndarray,
    stations: Sequence[int],
    baseline_last: Optional[np.ndarray] = None,
    horizon_weights: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> Dict[int, Dict[str, Any]]:
    """Calcula métricas de desempenho por estação."""
    B, T, S = preds.shape

    if horizon_weights is None:
        horizon_weights = np.ones(T, dtype=np.float64)
    horizon_weights = horizon_weights / (horizon_weights.sum() + eps)

    metrics: Dict[int, Dict[str, Any]] = {}

    for st_idx, station in enumerate(stations):
        y_pred = preds[:, :, st_idx].astype(np.float64)
        y_true = obs[:, :, st_idx].astype(np.float64)

        mask = ~np.isnan(y_true)
        y_pred_flat = y_pred[mask]
        y_true_flat = y_true[mask]

        # Overall metrics
        err_flat = y_pred_flat - y_true_flat
        rmse_overall = float(np.sqrt(np.mean(err_flat**2)))
        mae_overall = float(np.mean(np.abs(err_flat)))

        mape_mask_flat = np.abs(y_true_flat) > eps
        mape_overall = float(
            np.mean(np.abs(err_flat[mape_mask_flat]) / (np.abs(y_true_flat[mape_mask_flat]) + eps))
        ) * 100.0

        mu_true = float(np.mean(y_true_flat))
        ss_res = float(np.sum(err_flat**2))
        ss_tot = float(np.sum((y_true_flat - mu_true)**2))
        r2_overall = float(1.0 - ss_res / (ss_tot + eps))
        nse_overall = float(1.0 - ss_res / (ss_tot + eps))

        mu_pred = float(np.mean(y_pred_flat))
        std_true = float(np.std(y_true_flat) + eps)
        std_pred = float(np.std(y_pred_flat) + eps)
        cov = float(np.mean((y_true_flat - mu_true) * (y_pred_flat - mu_pred)))
        r = cov / (std_true * std_pred + eps)
        alpha = std_pred / (std_true + eps)
        beta = mu_pred / (mu_true + eps)
        kge_overall = float(1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2))

        skill_rmse_overall = None
        skill_rmse_h1 = None
        if baseline_last is not None:
            base_vec = baseline_last[:, st_idx].astype(np.float64)
            baseline = np.tile(base_vec[:, None], (1, T))
            base_flat = baseline[mask]
            rmse_base = float(np.sqrt(np.mean((base_flat - y_true_flat)**2)))
            skill_rmse_overall = float(1.0 - rmse_overall / (rmse_base + eps))
            
            # Skill RMSE para horizonte 1 (D+1)
            if T >= 1:
                y_true_h1 = y_true[:, 0][~np.isnan(y_true[:, 0])]
                pred_h1 = y_pred[:, 0][~np.isnan(y_true[:, 0])]
                base_h1 = base_vec[:len(y_true_h1)]
                
                if len(y_true_h1) > 0:
                    rmse_h1 = float(np.sqrt(np.mean((pred_h1 - y_true_h1)**2)))
                    rmse_base_h1 = float(np.sqrt(np.mean((base_h1 - y_true_h1)**2)))
                    skill_rmse_h1 = float(1.0 - rmse_h1 / (rmse_base_h1 + eps))

        # Per horizon
        rmse_t, mae_t, mape_t, r2_t, nse_t = [], [], [], [], []

        for t in range(T):
            mt = ~np.isnan(y_true[:, t])
            yt = y_true[mt, t]
            yp = y_pred[mt, t]

            if yt.size == 0:
                rmse_t.append(np.nan)
                mae_t.append(np.nan)
                mape_t.append(np.nan)
                r2_t.append(np.nan)
                nse_t.append(np.nan)
                continue

            e = yp - yt
            rmse_t.append(float(np.sqrt(np.mean(e**2))))
            mae_t.append(float(np.mean(np.abs(e))))

            mask_mape = np.abs(yt) > eps
            mape_t.append(
                float(np.mean(np.abs(e[mask_mape]) / (np.abs(yt[mask_mape]) + eps))) * 100.0
            )

            mu_yt = float(np.mean(yt))
            ss_res_t = float(np.sum(e**2))
            ss_tot_t = float(np.sum((yt - mu_yt)**2))
            r2_t.append(float(1.0 - ss_res_t / (ss_tot_t + eps)))
            nse_t.append(float(1.0 - ss_res_t / (ss_tot_t + eps)))

        rmse_t = np.array(rmse_t, dtype=np.float64)
        mae_t = np.array(mae_t, dtype=np.float64)
        mape_t = np.array(mape_t, dtype=np.float64)
        r2_t = np.array(r2_t, dtype=np.float64)
        nse_t = np.array(nse_t, dtype=np.float64)

        # Macro
        macro_rmse = float(np.nansum(rmse_t * horizon_weights))
        macro_mae = float(np.nansum(mae_t * horizon_weights))
        macro_mape = float(np.nansum(mape_t * horizon_weights))
        macro_r2 = float(np.nanmean(r2_t))
        macro_nse = float(np.nanmean(nse_t))

        metrics[station] = {
            "overall": {
                "rmse": rmse_overall,
                "mae": mae_overall,
                "mape": mape_overall,
                "r2": r2_overall,
                "nse": nse_overall,
                "kge": kge_overall,
                "skill_rmse": skill_rmse_overall,
                "skill_rmse_h1": skill_rmse_h1,
            },
            "macro": {
                "rmse": macro_rmse,
                "mae": macro_mae,
                "mape": macro_mape,
                "r2": macro_r2,
                "nse": macro_nse,
            },
            "per_horizon": {
                "rmse": rmse_t.tolist(),
                "mae": mae_t.tolist(),
                "mape": mape_t.tolist(),
                "r2": r2_t.tolist(),
                "nse": nse_t.tolist(),
            },
        }

    return metrics


def compute_metrics_by_event_type(
    preds: np.ndarray,
    obs: np.ndarray,
    stations: List[int],
    window_indices: Union[Dict[str, Dict[int, List[int]]], List[int], np.ndarray],
    baseline_last: Optional[np.ndarray] = None,
    horizon_weights: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Calcula métricas separadas por tipo de evento.
    
    window_indices pode ser:
      - Dict {event_type: {station: [indices]}} (padrão do analyze_flow_extremes)
      - List[int] ou np.ndarray (índices simples para todas as estações)
    """
    metrics_by_type = {}
    
    # ← NOVO: Tratamento de lista simples (índices de janelas)
    if isinstance(window_indices, (list, np.ndarray)):
        indices_list = list(window_indices)
        window_indices = {'all': {station: indices_list for station in stations}}
    
    # Verificar se é dicionário válido
    if not isinstance(window_indices, dict):
        raise ValueError(
            f"window_indices deve ser dict, list ou np.ndarray. "
            f"Recebido: {type(window_indices)}"
        )
    
    # Processar cada tipo de evento
    for event_type, indices_dict in window_indices.items():
        print(f"🔍 Calculando métricas para eventos: {event_type.upper()}")

        # indices_dict pode ser dict {station: [indices]} ou lista simples
        if isinstance(indices_dict, dict):
            station_indices = indices_dict
        elif isinstance(indices_dict, (list, np.ndarray)):
            station_indices = {station: list(indices_dict) for station in stations}
        else:
            print(f"  ⚠️  Tipo inválido para {event_type}, pulando")
            continue

        event_metrics = {}

        for st_idx, station in enumerate(stations):
            indices = station_indices.get(station, [])

            if len(indices) == 0:
                print(f"  ⚠️  Estação {station}: 0 janelas, pulando")
                continue

            # ← NOVO: Filtrar índices válidos (dentro do range)
            valid_indices = [i for i in indices if 0 <= i < preds.shape[0]]
            
            if len(valid_indices) == 0:
                print(f"  ⚠️  Estação {station}: nenhum índice válido, pulando")
                continue

            preds_filtered = preds[valid_indices, :, st_idx:st_idx+1]
            obs_filtered = obs[valid_indices, :, st_idx:st_idx+1]

            baseline_filtered = None
            if baseline_last is not None:
                baseline_filtered = baseline_last[valid_indices, st_idx:st_idx+1]

            station_metrics = compute_flow_metrics(
                preds=preds_filtered,
                obs=obs_filtered,
                stations=[station],
                baseline_last=baseline_filtered,
                horizon_weights=horizon_weights,
                eps=eps
            )

            station_metrics[station]['n_windows'] = len(valid_indices)
            event_metrics[station] = station_metrics[station]
            print(f"  ✅ Estação {station}: {len(valid_indices)} janelas processadas")

        metrics_by_type[event_type] = event_metrics

    return metrics_by_type


def save_metrics(
    metrics: Dict[str, Any],
    experiment_name: str,
    filename_base: str = "metrics",
    base_dir: Optional[str] = None,
    save_json: bool = True,
    save_csv: bool = True,
) -> Dict[str, str]:
    """
    Salva métricas em JSON e 2 arquivos CSV:
    
    1. overall.csv - Overall + Macro + Per_horizon para TODOS OS EVENTOS (agregado)
    2. by_event.csv - Overall + Macro + Per_horizon POR CADA EVENTO
    """
    try:
        from src.utils.experiment_utils import get_experiment_path
        exp_path = get_experiment_path(experiment_name)
    except ImportError:
        exp_path = Path("outputs/experiments") / experiment_name

    if base_dir is not None:
        exp_path = Path(base_dir) / experiment_name

    metrics_dir = exp_path / "predictions_test" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    first_key = next(iter(metrics.keys()), None)
    is_event_based = first_key in ['extreme', 'moderate', 'normal', 'extreme_high',
                                    'extreme_low', 'moderate_high', 'moderate_low', 'all']

    if is_event_based:
        event_types = list(metrics.keys())
    else:
        event_types = ['overall']
        metrics = {'overall': metrics}

    # ======== JSON COMPLETO ========
    if save_json:
        json_path = metrics_dir / f"{filename_base}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, default=str)
        saved_paths["json"] = str(json_path)
        print(f"💾 Métricas salvas (JSON): {json_path}")

    # ======== CSVs - apenas 2 arquivos ========
    if save_csv:
        # ========== 1. OVERALL.CSV - Agregado de todos os eventos ==========
        rows_all = []
        
        for event_label in event_types:
            if event_label not in metrics:
                continue
            station_metrics = metrics[event_label]

            for station, m in station_metrics.items():
                if not isinstance(m, dict):
                    continue

                # Overall
                if 'overall' in m:
                    overall = m['overall']
                    row = {
                        'event_type': event_label,
                        'station': station,
                        'metric_level': 'overall',
                        'rmse': overall.get('rmse'),
                        'mae': overall.get('mae'),
                        'mape': overall.get('mape'),
                        'r2': overall.get('r2'),
                        'nse': overall.get('nse'),
                        'kge': overall.get('kge'),
                        'skill_rmse': overall.get('skill_rmse'),
                        'n_windows': m.get('n_windows'),
                    }
                    rows_all.append(row)

                # Macro
                if 'macro' in m:
                    macro = m['macro']
                    row = {
                        'event_type': event_label,
                        'station': station,
                        'metric_level': 'macro',
                        'rmse': macro.get('rmse'),
                        'mae': macro.get('mae'),
                        'mape': macro.get('mape'),
                        'r2': macro.get('r2'),
                        'nse': macro.get('nse'),
                        'kge': None,
                        'skill_rmse': None,
                        'n_windows': m.get('n_windows'),
                    }
                    rows_all.append(row)

                # Per horizon
                if 'per_horizon' in m:
                    per_horizon = m['per_horizon']
                    if isinstance(per_horizon, dict):
                        n_horizons = len(per_horizon.get('rmse', []))
                        for h in range(n_horizons):
                            row = {
                                'event_type': event_label,
                                'station': station,
                                'metric_level': f'horizon_{h+1}',
                                'rmse': per_horizon.get('rmse', [None])[h],
                                'mae': per_horizon.get('mae', [None])[h],
                                'mape': per_horizon.get('mape', [None])[h],
                                'r2': per_horizon.get('r2', [None])[h],
                                'nse': per_horizon.get('nse', [None])[h],
                                'kge': None,
                                'skill_rmse': None,
                                'n_windows': m.get('n_windows'),
                            }
                            rows_all.append(row)

        if rows_all:
            df_all = pd.DataFrame(rows_all)
            csv_overall_path = metrics_dir / f"{filename_base}_overall.csv"
            df_all.to_csv(csv_overall_path, index=False, sep='\t')
            saved_paths["csv_overall"] = str(csv_overall_path)
            print(f"💾 Métricas salvas (CSV Overall): {csv_overall_path}")

        # ========== 2. BY_EVENT.CSV - Separado por evento ==========
        rows_by_event = []
        
        for event_label in event_types:
            if event_label not in metrics:
                continue
            station_metrics = metrics[event_label]

            for station, m in station_metrics.items():
                if not isinstance(m, dict):
                    continue

                # Overall
                if 'overall' in m:
                    overall = m['overall']
                    row = {
                        'event_type': event_label,
                        'station': station,
                        'metric_level': 'overall',
                        'rmse': overall.get('rmse'),
                        'mae': overall.get('mae'),
                        'mape': overall.get('mape'),
                        'r2': overall.get('r2'),
                        'nse': overall.get('nse'),
                        'kge': overall.get('kge'),
                        'skill_rmse': overall.get('skill_rmse'),
                        'n_windows': m.get('n_windows'),
                    }
                    rows_by_event.append(row)

                # Macro
                if 'macro' in m:
                    macro = m['macro']
                    row = {
                        'event_type': event_label,
                        'station': station,
                        'metric_level': 'macro',
                        'rmse': macro.get('rmse'),
                        'mae': macro.get('mae'),
                        'mape': macro.get('mape'),
                        'r2': macro.get('r2'),
                        'nse': macro.get('nse'),
                        'kge': None,
                        'skill_rmse': None,
                        'n_windows': m.get('n_windows'),
                    }
                    rows_by_event.append(row)

                # Per horizon
                if 'per_horizon' in m:
                    per_horizon = m['per_horizon']
                    if isinstance(per_horizon, dict):
                        n_horizons = len(per_horizon.get('rmse', []))
                        for h in range(n_horizons):
                            row = {
                                'event_type': event_label,
                                'station': station,
                                'metric_level': f'horizon_{h+1}',
                                'rmse': per_horizon.get('rmse', [None])[h],
                                'mae': per_horizon.get('mae', [None])[h],
                                'mape': per_horizon.get('mape', [None])[h],
                                'r2': per_horizon.get('r2', [None])[h],
                                'nse': per_horizon.get('nse', [None])[h],
                                'kge': None,
                                'skill_rmse': None,
                                'n_windows': m.get('n_windows'),
                            }
                            rows_by_event.append(row)

        if rows_by_event:
            df_by_event = pd.DataFrame(rows_by_event)
            csv_by_event_path = metrics_dir / f"{filename_base}_by_event.csv"
            df_by_event.to_csv(csv_by_event_path, index=False, sep='\t')
            saved_paths["csv_by_event"] = str(csv_by_event_path)
            print(f"💾 Métricas salvas (CSV By Event): {csv_by_event_path}")

    return saved_paths


def print_metrics_summary(metrics: Dict[int, Dict[str, Any]]) -> None:
    """Imprime resumo formatado das métricas."""
    print("" + "="*80)
    print("RESUMO DAS MÉTRICAS POR ESTAÇÃO")
    print("="*80)

    for station, station_metrics in metrics.items():
        print(f"📍 Estação {station}")
        print("-" * 60)

        overall = station_metrics["overall"]
        print("  Overall:")
        print(f"    RMSE:       {overall['rmse']:.3f} m³/s")
        print(f"    MAE:        {overall['mae']:.3f} m³/s")
        print(f"    MAPE:       {overall['mape']:.2f}%")
        print(f"    R²:         {overall['r2']:.4f}")
        print(f"    NSE:        {overall['nse']:.4f}")
        print(f"    KGE:        {overall['kge']:.4f}")
        if overall['skill_rmse'] is not None:
            print(f"    Skill RMSE: {overall['skill_rmse']:.4f}")

        macro = station_metrics["macro"]
        print("Macro (ponderado por horizonte):")
        print(f"    RMSE:       {macro['rmse']:.3f} m³/s")
        print(f"    MAE:        {macro['mae']:.3f} m³/s")
        print(f"    MAPE:       {macro['mape']:.2f}%")
        print(f"    R²:         {macro['r2']:.4f}")
        print(f"    NSE:        {macro['nse']:.4f}")

    print("" + "="*80)


def print_metrics_comparison_by_event(
    metrics_by_type: Dict[str, Dict[int, Dict[str, Any]]],
    stations: List[int]
) -> None:
    """Imprime comparação de métricas entre tipos de eventos."""
    print("" + "="*80)
    print("COMPARAÇÃO DE MÉTRICAS POR TIPO DE EVENTO")
    print("="*80)

    for station in stations:
        print(f"📍 Estação {station}")
        print("-" * 60)
        print(f"{'Tipo':<12} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'NSE':>10} {'KGE':>10}")
        print("-" * 60)

        for event_type in ['extreme', 'moderate', 'normal']:
            if event_type not in metrics_by_type:
                continue

            if station not in metrics_by_type[event_type]:
                print(f"{event_type:<12} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
                continue

            overall = metrics_by_type[event_type][station]['overall']
            print(f"{event_type:<12} "
                  f"{overall['rmse']:>10.3f} "
                  f"{overall['mae']:>10.3f} "
                  f"{overall['r2']:>10.4f} "
                  f"{overall['nse']:>10.4f} "
                  f"{overall['kge']:>10.4f}")

    print("" + "="*80)
