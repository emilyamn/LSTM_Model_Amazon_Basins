"""
Módulo para cálculo de métricas de desempenho do modelo.
"""

from typing import Dict, Any, Optional, Sequence
import numpy as np


def compute_flow_metrics(
    preds: np.ndarray,
    obs: np.ndarray,
    stations: Sequence[int],
    baseline_last: Optional[np.ndarray] = None,
    horizon_weights: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> Dict[int, Dict[str, Any]]:
    """
    Calcula métricas de desempenho por estação.

    Args:
        preds: Previsões, shape (batch, horizon, n_stations)
        obs: Observações, shape (batch, horizon, n_stations)
        stations: Lista de IDs das estações
        baseline_last: Baseline de persistência, shape (batch, n_stations)
        horizon_weights: Pesos por horizonte, shape (horizon,)
        eps: Epsilon para evitar divisão por zero

    Returns:
        Dicionário com métricas por estação:
            - overall: métricas gerais (rmse, mae, mape, r2, nse, kge, skill_rmse)
            - macro: médias ponderadas por horizonte
            - per_horizon: arrays de métricas por horizonte
    """
    B, T, S = preds.shape

    if horizon_weights is None:
        horizon_weights = np.ones(T, dtype=np.float64)
    horizon_weights = horizon_weights / (horizon_weights.sum() + eps)

    metrics: Dict[int, Dict[str, Any]] = {}

    for st_idx, station in enumerate(stations):
        y_pred = preds[:, :, st_idx].astype(np.float64)
        y_true = obs[:, :, st_idx].astype(np.float64)

        # Máscara para valores válidos
        mask = ~np.isnan(y_true)
        y_pred_flat = y_pred[mask]
        y_true_flat = y_true[mask]

        # ==========================================
        # MÉTRICAS OVERALL (todas as previsões)
        # ==========================================
        err_flat = y_pred_flat - y_true_flat
        rmse_overall = float(np.sqrt(np.mean(err_flat**2)))
        mae_overall = float(np.mean(np.abs(err_flat)))

        mape_mask_flat = np.abs(y_true_flat) > eps
        mape_overall = float(
            np.mean(np.abs(err_flat[mape_mask_flat]) / (np.abs(y_true_flat[mape_mask_flat]) + eps))
        ) * 100.0

        # R² e NSE
        mu_true = float(np.mean(y_true_flat))
        ss_res = float(np.sum(err_flat**2))
        ss_tot = float(np.sum((y_true_flat - mu_true)**2))
        r2_overall = float(1.0 - ss_res / (ss_tot + eps))
        nse_overall = float(1.0 - ss_res / (ss_tot + eps))

        # KGE
        mu_pred = float(np.mean(y_pred_flat))
        std_true = float(np.std(y_true_flat) + eps)
        std_pred = float(np.std(y_pred_flat) + eps)
        cov = float(np.mean((y_true_flat - mu_true) * (y_pred_flat - mu_pred)))
        r = cov / (std_true * std_pred + eps)
        alpha = std_pred / (std_true + eps)
        beta = mu_pred / (mu_true + eps)
        kge_overall = float(1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2))

        # Skill vs persistência
        skill_rmse_overall = None
        if baseline_last is not None:
            base_vec = baseline_last[:, st_idx].astype(np.float64)
            baseline = np.tile(base_vec[:, None], (1, T))
            base_flat = baseline[mask]
            rmse_base = float(np.sqrt(np.mean((base_flat - y_true_flat)**2)))
            skill_rmse_overall = float(1.0 - rmse_overall / (rmse_base + eps))

        # ==========================================
        # MÉTRICAS POR HORIZONTE
        # ==========================================
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

        # ==========================================
        # MÉTRICAS MACRO (ponderadas)
        # ==========================================
        macro_rmse = float(np.nansum(rmse_t * horizon_weights))
        macro_mae = float(np.nansum(mae_t * horizon_weights))
        macro_mape = float(np.nansum(mape_t * horizon_weights))
        macro_r2 = float(np.nanmean(r2_t))
        macro_nse = float(np.nanmean(nse_t))

        # ==========================================
        # ARMAZENAR RESULTADOS
        # ==========================================
        metrics[station] = {
            "overall": {
                "rmse": rmse_overall,
                "mae": mae_overall,
                "mape": mape_overall,
                "r2": r2_overall,
                "nse": nse_overall,
                "kge": kge_overall,
                "skill_rmse": skill_rmse_overall,
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


def print_metrics_summary(metrics: Dict[int, Dict[str, Any]]) -> None:
    """
    Imprime resumo formatado das métricas.

    Args:
        metrics: Dicionário de métricas retornado por compute_flow_metrics
    """
    print("\n" + "="*80)
    print("RESUMO DAS MÉTRICAS POR ESTAÇÃO")
    print("="*80)

    for station, station_metrics in metrics.items():
        print(f"\n📍 Estação {station}")
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
        print("\n  Macro (ponderado por horizonte):")
        print(f"    RMSE:       {macro['rmse']:.3f} m³/s")
        print(f"    MAE:        {macro['mae']:.3f} m³/s")
        print(f"    MAPE:       {macro['mape']:.2f}%")
        print(f"    R²:         {macro['r2']:.4f}")
        print(f"    NSE:        {macro['nse']:.4f}")

    print("\n" + "="*80)
