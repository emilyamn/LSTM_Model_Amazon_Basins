"""
Funções para treino do modelo hidrológico.
"""
from typing import Optional, Tuple, Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.data_structures import Scaler # se for necessário
from ..training.losses import multi_step_loss
from ..utils.data_utils import move_sample_to_device


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    decoder_history: int,
    decoder_horizon: int,
    max_epochs: int = 30,
    initial_teacher_forcing: float = 0.60,
    teacher_forcing_decay: float = 0.94,
    final_teacher_forcing: float = 0.0,
    free_run_tail: int = 8,
    lambda_smooth: float = 0.001,
    lambda_negative: float = 0.01,
    lambda_continuity: float = 0.002,
    lambda_slope: float = 1.0,
    horizon_weight_mode: str = "increasing",
    horizon_weight_gamma: float = 0.03,
    early_free_run_patience: Optional[int] = None,
    lambda_gate_bias: float = 0.01,
    gate_decay: float = 0.06,
    lambda_direction: float = 0.02,
    direction_start: int = 5,
    dir_weight_gamma: float = 0.05,
    patience: int = 10,
    min_delta: float = 1e-4,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    clip_grad_norm: float = 3.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.nn.Module:
    """
    Treina o modelo Seq2SeqHydro.
    
    Args:
        model: Modelo a ser treinado
        train_loader: DataLoader de treino
        val_loader: DataLoader de validação
        decoder_history: Número de passos históricos no decoder
        decoder_horizon: Horizonte de previsão
        max_epochs: Número máximo de épocas
        initial_teacher_forcing: Teacher forcing inicial
        teacher_forcing_decay: Decaimento do teacher forcing
        final_teacher_forcing: Teacher forcing final
        free_run_tail: Épocas finais em free run
        lambda_smooth: Peso da regularização de suavidade
        lambda_negative: Peso da penalidade de negativos
        lambda_continuity: Peso da continuidade
        lambda_slope: Peso da penalidade de slope
        horizon_weight_mode: Modo de pesos por horizonte
        horizon_weight_gamma: Gamma para pesos por horizonte
        early_free_run_patience: Paciência para early free run
        lambda_gate_bias: Peso da penalidade do gate
        gate_decay: Decaimento do gate penalty
        lambda_direction: Peso da penalidade de direção
        direction_start: Passo inicial para penalidade de direção
        dir_weight_gamma: Gamma para pesos de direção
        patience: Paciência para early stopping
        min_delta: Delta mínimo para melhoria
        learning_rate: Taxa de aprendizado
        weight_decay: Decaimento de peso
        clip_grad_norm: Norma máxima para clipping de gradiente
        device: Dispositivo para treino
        
    Returns:
        Modelo treinado
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.7)

    # Configurar free run tail
    free_run_tail = max(0, min(free_run_tail, max_epochs - 1))

    # Pesos por horizonte
    if horizon_weight_mode == "increasing":
        weights = torch.exp(horizon_weight_gamma * torch.arange(decoder_horizon, device=device))
    else:
        weights = torch.ones(decoder_horizon, device=device)

    # Inicialização para early stopping
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # Calcular teacher forcing ratio
        if epoch > max_epochs - free_run_tail:
            tf_ratio = final_teacher_forcing
        else:
            tf_ratio = max(
                final_teacher_forcing, 
                initial_teacher_forcing * (teacher_forcing_decay ** (epoch - 1))
            )
        
        if (early_free_run_patience is not None) and (no_improve >= early_free_run_patience):
            tf_ratio = 0.0

        # Reduzir continuidade com o tempo
        lambda_cont_epoch = lambda_continuity * (0.95 ** (epoch - 1))

        # Treino
        model.train()
        train_losses = []
        
        for batch in train_loader:
            batch = move_sample_to_device(batch, device)
            
            preds, _, g_seq = model(batch, tf_ratio, decoder_history, decoder_horizon)
            
            loss = multi_step_loss(
                preds, 
                batch.target, 
                batch.baseline_last, 
                weights,
                lambda_smooth, 
                lambda_negative, 
                lambda_cont_epoch,
                use_huber=False,
                g_seq=g_seq, 
                lambda_gate_bias=lambda_gate_bias, 
                gate_decay=gate_decay,
                lambda_direction=lambda_direction, 
                direction_start=direction_start, 
                dir_weight_gamma=dir_weight_gamma,
                lambda_slope=lambda_slope
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))

        # Validação
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = move_sample_to_device(batch, device)
                preds, _, g_seq = model(batch, 0.0, decoder_history, decoder_horizon)
                
                loss = multi_step_loss(
                    preds, 
                    batch.target, 
                    batch.baseline_last, 
                    weights,
                    lambda_smooth, 
                    lambda_negative, 
                    lambda_cont_epoch,
                    use_huber=False,
                    g_seq=g_seq, 
                    lambda_gate_bias=lambda_gate_bias, 
                    gate_decay=gate_decay,
                    lambda_direction=lambda_direction, 
                    direction_start=direction_start, 
                    dir_weight_gamma=dir_weight_gamma,
                    lambda_slope=lambda_slope
                )
                val_losses.append(loss.item())

        avg_val = float(np.mean(val_losses))
        scheduler.step(avg_val)

        # Early stopping
        improved = avg_val < (best_val - min_delta)
        if improved:
            best_val = avg_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        print(f"[Epoch {epoch:02d}] train={avg_train:.4f} val={avg_val:.4f} tf={tf_ratio:.3f}")
        
        if no_improve >= patience:
            print("Early stopping.")
            break

    # Carregar melhor modelo
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def predict_autoregressive(
    model,
    loader: DataLoader,
    decoder_history: int,
    decoder_horizon: int,
    scalers: Dict[str, Scaler],
    stations: List[int],
    clamp_non_negative: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, str]:
    """
    Realiza predições autoregressivas.
    
    Args:
        model: Modelo treinado
        loader: DataLoader para predição
        decoder_history: Número de passos históricos no decoder
        decoder_horizon: Horizonte de previsão
        scalers: Escaladores para as estações
        stations: Lista de IDs das estações
        clamp_non_negative: Forçar valores não negativos
        device: Dispositivo para predição
        
    Returns:
        Tupla com (previsões, observações, baseline, gates, datas, dispositivo)
    """
    preds_all, obs_all = [], []
    baseline_batches, gseq_batches = [], []
    forecast_dates_batches = []
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_device = move_sample_to_device(batch, device)
            preds, _, g_seq = model(batch_device, 0.0, decoder_history, decoder_horizon)
            
            preds_all.append(preds.cpu().numpy())
            obs_all.append(batch_device.target.cpu().numpy())
            baseline_batches.append(batch_device.baseline_last.cpu().numpy())
            
            if g_seq is not None:
                gseq_batches.append(g_seq.cpu().numpy())
            
            forecast_dates_batches.append(batch.forecast_date)

    # Concatenar todos os batches
    preds_all = np.concatenate(preds_all, axis=0)
    obs_all = np.concatenate(obs_all, axis=0)
    baseline_last_all = np.concatenate(baseline_batches, axis=0)
    g_seq_all = np.concatenate(gseq_batches, axis=0) if len(gseq_batches) > 0 else None
    forecast_dates_all = np.concatenate(forecast_dates_batches, axis=0)

    # Desscalar previsões
    for st_idx, station in enumerate(stations):
        scaler = scalers[f"Q_{station}"]
        preds_all[:, :, st_idx] = scaler.inverse_transform(
            torch.from_numpy(preds_all[:, :, st_idx])
        ).numpy()
        obs_all[:, :, st_idx] = scaler.inverse_transform(
            torch.from_numpy(obs_all[:, :, st_idx])
        ).numpy()
        baseline_last_all[:, st_idx] = scaler.inverse_transform(
            torch.from_numpy(baseline_last_all[:, st_idx])
        ).numpy()

    # Forçar valores não negativos
    if clamp_non_negative:
        preds_all = np.clip(preds_all, 0.0, None)
        baseline_last_all = np.clip(baseline_last_all, 0.0, None)

    return preds_all, obs_all, baseline_last_all, g_seq_all, forecast_dates_all, device