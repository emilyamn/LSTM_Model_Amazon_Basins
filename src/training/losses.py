"""
Funções de loss para treino do modelo hidrológico.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def multi_step_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    baseline_last: torch.Tensor,
    weights: torch.Tensor,
    lambda_smooth: float,
    lambda_negative: float,
    lambda_continuity: float,
    use_huber: bool = False,
    huber_delta: float = 1.0,
    g_seq: Optional[torch.Tensor] = None,
    lambda_gate_bias: float = 0.01,
    gate_decay: float = 0.12,
    lambda_direction: float = 0.02,
    direction_start: int = 5,
    dir_weight_gamma: float = 0.05,
    lambda_slope: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Loss multi-step para previsão de séries temporais.
    
    Args:
        preds: Previsões do modelo
        target: Valores reais
        baseline_last: Última observação (persistência)
        weights: Pesos por horizonte
        lambda_smooth: Peso da regularização de suavidade
        lambda_negative: Peso da penalidade de negativos
        lambda_continuity: Peso da continuidade
        use_huber: Usar Huber loss em vez de MSE
        huber_delta: Delta para Huber loss
        g_seq: Sequência de gates
        lambda_gate_bias: Peso da penalidade do gate
        gate_decay: Decaimento do gate penalty
        lambda_direction: Peso da penalidade de direção
        direction_start: Passo inicial para penalidade de direção
        dir_weight_gamma: Gamma para pesos de direção
        lambda_slope: Peso da penalidade de slope
        mask: Máscara de valores válidos
        
    Returns:
        Valor da loss
    """
    eps = 1e-8
    if mask is None:
        mask = torch.ones_like(preds)

    # 1. Termo de dados (MSE ou Huber)
    if use_huber:
        data_loss_all = F.huber_loss(preds, target, reduction="none", delta=huber_delta)
    else:
        data_loss_all = (preds - target) ** 2  # MSE
        
    data_loss = (data_loss_all * weights.view(1, -1, 1) * mask).sum() / (mask.sum() + eps)

    # 2. Slope Loss (penaliza erro na taxa de variação)
    if preds.shape[1] > 1:
        slope_pred = preds[:, 1:, :] - preds[:, :-1, :]
        slope_true = target[:, 1:, :] - target[:, :-1, :]
        slope_mask = mask[:, 1:, :] * mask[:, :-1, :]
        slope_loss = (torch.abs(slope_pred - slope_true) * slope_mask).sum() / (slope_mask.sum() + eps)
    else:
        slope_loss = preds.new_tensor(0.0)

    # 3. Penalidade de valores negativos
    neg_penalty = (F.relu(-preds) * mask).sum() / (mask.sum() + eps)

    # 4. Regularização de suavidade
    if preds.shape[1] > 3:
        diff_long = torch.abs(preds[:, 3:, :] - preds[:, :-3, :])
        mask_long = mask[:, 3:, :] * mask[:, :-3, :]
        smooth_penalty = (diff_long * mask_long).sum() / (mask_long.sum() + eps)
    else:
        smooth_penalty = preds.new_tensor(0.0)

    # 5. Continuidade no primeiro passo
    delta_pred0 = preds[:, 0, :] - baseline_last
    delta_true0 = target[:, 0, :] - baseline_last
    mask0 = mask[:, 0, :]
    continuity_penalty = ((delta_pred0 - delta_true0) ** 2 * mask0).sum() / (mask0.sum() + eps)

    # 6. Penalidade de direção
    if preds.size(1) >= 2:
        dp = preds[:, 1:, :] - preds[:, :-1, :]
        dt = target[:, 1:, :] - target[:, :-1, :]
        m2 = mask[:, 1:, :] * mask[:, :-1, :]
        dir_term = F.relu(-(dp * dt)) / (torch.abs(dt) + eps)
        t_idx2 = torch.arange(dp.size(1), device=preds.device).view(1, -1, 1)
        w_dir = torch.exp(dir_weight_gamma * t_idx2) * (t_idx2 >= direction_start).float()
        direction_penalty = (dir_term * m2 * w_dir).sum() / (m2.sum() + eps)
    else:
        direction_penalty = preds.new_tensor(0.0)

    # 7. Penalidade do gate
    if g_seq is not None:
        gate_penalty = (g_seq ** 2).mean()  # Queremos gate pequeno (perto de 0)
    else:
        gate_penalty = preds.new_tensor(0.0)

    # Loss total
    total_loss = (
        data_loss
        + lambda_slope * slope_loss
        + lambda_smooth * smooth_penalty
        + lambda_negative * neg_penalty
        + lambda_continuity * continuity_penalty
        + lambda_gate_bias * gate_penalty
        + lambda_direction * direction_penalty
    )
    
    return total_loss