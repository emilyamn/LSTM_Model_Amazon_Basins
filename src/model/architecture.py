"""
Arquitetura do modelo Seq2Seq para previsão hidrológica.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from src.data.data_structures import Sample
from src.model.layers import StaticEmbedding


class Seq2SeqHydro(nn.Module):
    """Modelo Seq2Seq para previsão de vazões."""

    def __init__(
        self,
        encoder_input_dim: int,
        decoder_input_dim: int,
        n_static: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        n_stations: int,
        attention: bool = True,
        residual: bool = True,
        non_negative: bool = True,
        input_noise_std: float = 0.05,
        y_prev_dropout_p: float = 0.30,
        gate_y_prev: bool = True,
        tf_step_decay: float = 0.15,
        y_prev_mask_p: float = 0.30,
        decoder_feat_dropout_p: float = 0.15,
        gate_from_inputs: bool = True,
        clamp_gate_by_ceiling: bool = True,
        gate_ceiling_min: float = 0.10,
        detach_y_prev_in_gate: bool = True,
        y_prev_mask_step_gamma: float = 0.20,
        gate_min: float = 0.0,
        gate_max: float = 0.60,
        n_decoder_flow_feats: int = 0,
    ):
        """
        Inicializa o modelo Seq2SeqHydro.

        Args:
            encoder_input_dim: Dimensão de entrada do encoder
            decoder_input_dim: Dimensão de entrada do decoder
            n_static: Número de features estáticas
            hidden_dim: Dimensão do espaço oculto
            num_layers: Número de camadas LSTM
            dropout: Taxa de dropout
            n_stations: Número de estações
            attention: Usar atenção
            residual: Usar conexões residuais
            non_negative: Forçar saídas não negativas
            input_noise_std: Desvio padrão do ruído de entrada
            y_prev_dropout_p: Dropout da observação anterior
            gate_y_prev: Usar gate para observação anterior
            tf_step_decay: Decaimento do teacher forcing
            y_prev_mask_p: Probabilidade de mascarar y_prev
            decoder_feat_dropout_p: Dropout das features do decoder
            gate_from_inputs: Gate baseado nas inputs
            clamp_gate_by_ceiling: Limitar gate por teto
            gate_ceiling_min: Valor mínimo do teto do gate
            detach_y_prev_in_gate: Desacoplar y_prev no gate
            y_prev_mask_step_gamma: Gamma do mascaramento step
            gate_min: Valor mínimo do gate
            gate_max: Valor máximo do gate
        """
        super().__init__()
        self.n_stations = n_stations
        self.hidden_dim = hidden_dim
        self.attention = attention
        self.num_layers = num_layers
        self.residual = residual
        self.non_negative = non_negative
        self.input_noise_std = input_noise_std
        self.tf_step_decay = tf_step_decay
        self.gate_y_prev = gate_y_prev
        self.y_prev_mask_p = y_prev_mask_p
        self.clamp_gate_by_ceiling = clamp_gate_by_ceiling
        self.gate_ceiling_min = gate_ceiling_min
        self.detach_y_prev_in_gate = detach_y_prev_in_gate
        self.y_prev_mask_step_gamma = y_prev_mask_step_gamma
        self.dec_in_dim = decoder_input_dim
        self.gate_from_inputs = gate_from_inputs
        self.gate_min = gate_min
        self.gate_max = gate_max
        self.n_decoder_flow_feats = n_decoder_flow_feats

        # Encoder LSTM
        self.encoder = nn.LSTM(
            encoder_input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder LSTM — SEM y_prev no input (forçar aprendizado de clima)
        # decoder_input_dim inclui n_stations (y_prev), mas o LSTM recebe sem y_prev
        self.decoder_lstm_dim = decoder_input_dim - n_stations
        self.decoder = nn.LSTM(
            self.decoder_lstm_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Camada de atenção
        if attention:
            self.attn_layer = nn.MultiheadAttention(
                hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )

        # Normalização e embedding estático
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.static_embed = StaticEmbedding(n_static, hidden_dim * num_layers)

        # Camada de saída (LSTM pathway)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_stations),
        )

        # Climate skip connection — caminho direto de features climáticas para delta_t
        # Exclui features de fluxo (n_decoder_flow_feats) do input da climate_proj
        climate_feat_dim = decoder_input_dim - n_stations - n_decoder_flow_feats
        self.climate_proj = nn.Sequential(
            nn.Linear(climate_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_stations),
        )

        # Dropouts
        self.y_prev_dropout = nn.Dropout(p=y_prev_dropout_p)
        self.decoder_in_dropout = nn.Dropout(p=decoder_feat_dropout_p)

        # MLP para gate
        gate_in_dim = (decoder_input_dim - n_stations) if gate_from_inputs else hidden_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_stations),
        )

    def forward(
        self,
        sample: Sample,
        teacher_forcing_ratio: float,
        decoder_history: int,
        decoder_horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass do modelo.

        Args:
            sample: Objeto Sample com dados de entrada
            teacher_forcing_ratio: Razão de teacher forcing
            decoder_history: Número de passos históricos no decoder
            decoder_horizon: Horizonte de previsão

        Returns:
            Tupla com (previsões, máscara, gates)
        """
        batch_size = sample.encoder_dyn.size(0)
        device = sample.encoder_dyn.device

        # Extrair tensores do sample
        enc_in = sample.encoder_dyn
        dec_dyn = sample.decoder_dyn
        dec_temp = sample.temporal_dec
        static_feats = sample.static
        target = sample.target.to(device)

        # Embedding estático
        static_embed = self.static_embed(static_feats)
        static_embed = static_embed.view(-1, self.num_layers, self.hidden_dim)
        static_embed = static_embed.permute(1, 0, 2).contiguous()

        # Encoder
        enc_out, (h, c) = self.encoder(enc_in, (static_embed, static_embed))
        attn_memory = enc_out if self.attention else None

        # Loop de previsão autoregressiva
        preds: List[torch.Tensor] = []
        g_steps: List[torch.Tensor] = []
        y_prev = sample.baseline_last

        # Adicionar ruído durante treino
        if self.training and self.input_noise_std > 0.0:
            y_prev = y_prev + torch.randn_like(y_prev) * self.input_noise_std

        for t in range(decoder_horizon):
            idx = decoder_history + t
            tf_t = float(teacher_forcing_ratio * math.exp(-self.tf_step_decay * t))

            # Features externas (clima + temporal) — SEM y_prev no LSTM
            ext_features = torch.cat([dec_dyn[:, idx, :], dec_temp[:, idx, :]], dim=-1)
            dec_in_t = ext_features.unsqueeze(1)

            if self.training:
                dec_in_t = self.decoder_in_dropout(dec_in_t)

            # Decoder LSTM
            dec_out_t, (h, c) = self.decoder(dec_in_t, (h, c))

            # Atenção
            if self.attention:
                attn_out, _ = self.attn_layer(dec_out_t, attn_memory, attn_memory)
                dec_out_t = dec_out_t + attn_out

            dec_out_t = self.layernorm(dec_out_t.squeeze(1))

            # Previsão: LSTM pathway + climate skip connection
            delta_lstm = self.out_proj(dec_out_t)
            # climate_proj recebe apenas features climáticas (sem fluxo)
            climate_only = ext_features[:, self.n_decoder_flow_feats:]
            delta_climate = self.climate_proj(climate_only)
            delta_t = delta_lstm + delta_climate
            base_pred = y_prev + delta_t if self.residual else delta_t

            # Gate mechanism
            if self.gate_y_prev:
                gate_inputs = ext_features
                if self.detach_y_prev_in_gate:
                    gate_inputs = gate_inputs.detach()

                g_raw = torch.sigmoid(self.gate_mlp(gate_inputs))

                if self.clamp_gate_by_ceiling:
                    ratio = t / max(1, (decoder_horizon - 1))
                    g_max_t = self.gate_max - (self.gate_max - self.gate_ceiling_min) * ratio
                    g_max_t = max(self.gate_min + 1e-3, g_max_t)
                else:
                    g_max_t = self.gate_max

                g = self.gate_min + (g_max_t - self.gate_min) * g_raw
                pred_t = g * y_prev + (1.0 - g) * base_pred
                g_steps.append(g.unsqueeze(1))
            else:
                pred_t = base_pred

            # Forçar não negatividade
            if self.non_negative:
                pred_t = F.relu(pred_t)

            # Teacher forcing
            if self.training:
                use_teacher = (torch.rand(batch_size, 1, device=device) < tf_t).float()
                y_prev = use_teacher * target[:, t, :] + (1.0 - use_teacher) * pred_t
            else:
                y_prev = pred_t

            preds.append(pred_t.unsqueeze(1))

        # Concatenar previsões
        preds = torch.cat(preds, dim=1)
        g_seq = torch.cat(g_steps, dim=1) if len(g_steps) > 0 else None

        return preds, sample.mask_dec[:, decoder_history:, :], g_seq

    @torch.no_grad()
    def diagnostic_forward(
        self,
        sample: Sample,
        decoder_history: int,
        decoder_horizon: int,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass com captura de diagnósticos internos (modo inferência).

        Returns:
            Tupla com (previsões, diagnostics_dict)
        """
        self.eval()
        device = sample.encoder_dyn.device

        enc_in = sample.encoder_dyn
        dec_dyn = sample.decoder_dyn
        dec_temp = sample.temporal_dec
        static_feats = sample.static

        # Embedding estático
        static_embed = self.static_embed(static_feats)
        static_embed = static_embed.view(-1, self.num_layers, self.hidden_dim)
        static_embed = static_embed.permute(1, 0, 2).contiguous()

        # Encoder
        enc_out, (h, c) = self.encoder(enc_in, (static_embed, static_embed))
        attn_memory = enc_out if self.attention else None

        # Diagnósticos
        diag = {
            'delta_lstm': [],
            'delta_climate': [],
            'delta_total': [],
            'gate': [],
            'y_prev': [],
            'pred': [],
            'ext_features_mean': [],
            'ext_features_std': [],
        }

        preds: List[torch.Tensor] = []
        y_prev = sample.baseline_last

        for t in range(decoder_horizon):
            idx = decoder_history + t

            ext_features = torch.cat([dec_dyn[:, idx, :], dec_temp[:, idx, :]], dim=-1)
            dec_in_t = ext_features.unsqueeze(1)

            dec_out_t, (h, c) = self.decoder(dec_in_t, (h, c))

            if self.attention:
                attn_out, _ = self.attn_layer(dec_out_t, attn_memory, attn_memory)
                dec_out_t = dec_out_t + attn_out

            dec_out_t = self.layernorm(dec_out_t.squeeze(1))

            delta_lstm = self.out_proj(dec_out_t)
            climate_only = ext_features[:, self.n_decoder_flow_feats:]
            delta_climate = self.climate_proj(climate_only)
            delta_t = delta_lstm + delta_climate
            base_pred = y_prev + delta_t if self.residual else delta_t

            if self.gate_y_prev:
                gate_inputs = ext_features
                g_raw = torch.sigmoid(self.gate_mlp(gate_inputs))

                if self.clamp_gate_by_ceiling:
                    ratio = t / max(1, (decoder_horizon - 1))
                    g_max_t = self.gate_max - (self.gate_max - self.gate_ceiling_min) * ratio
                    g_max_t = max(self.gate_min + 1e-3, g_max_t)
                else:
                    g_max_t = self.gate_max

                g = self.gate_min + (g_max_t - self.gate_min) * g_raw
                pred_t = g * y_prev + (1.0 - g) * base_pred
            else:
                g = torch.zeros_like(delta_t)
                pred_t = base_pred

            if self.non_negative:
                pred_t = F.relu(pred_t)

            # Capturar diagnósticos
            diag['delta_lstm'].append(delta_lstm.detach())
            diag['delta_climate'].append(delta_climate.detach())
            diag['delta_total'].append(delta_t.detach())
            diag['gate'].append(g.detach())
            diag['y_prev'].append(y_prev.detach())
            diag['pred'].append(pred_t.detach())
            diag['ext_features_mean'].append(ext_features.mean(dim=-1).detach())
            diag['ext_features_std'].append(ext_features.std(dim=-1).detach())

            y_prev = pred_t
            preds.append(pred_t.unsqueeze(1))

        preds = torch.cat(preds, dim=1)
        return preds, diag