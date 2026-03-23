"""
Dataset hidrológico para treino de modelos de deep learning.
"""

from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, Subset
import torch
import numpy as np
import pandas as pd

from src.data.data_structures import Scaler, Sample, compute_scaler
from src.utils.time_utils import compute_time_axes


class HydroDataset(Dataset):
    """Dataset para dados hidrológicos com suporte a múltiplas estações."""

    def __init__(
        self,
        df: pd.DataFrame,
        stations: List[int],
        static_attrs: Dict[int, Dict[str, float]],
        train_indices: np.ndarray,
        forecast_cols: Dict[int, Tuple[str, str]],
        flow_window_config: Dict,
        climate_window_config: Dict,
        temporal_features: List[str],
        api_k_list: List[float],
        static_keys: List[str],
        window_stride: int = 1,
        reserve_last_days: int = 0,
        forcings: str = "P",  # ✅ NOVO PARÂMETRO
    ):
        self.df = df
        self.stations = stations
        self.static_attrs = static_attrs
        self.forecast_cols = forecast_cols
        self.flow_window_config = flow_window_config
        self.climate_window_config = climate_window_config
        self.temporal_features = temporal_features
        self.api_k_list = api_k_list
        self.static_keys = static_keys
        self.window_stride = window_stride
        self.reserve_last_days = reserve_last_days
        self.forcings = forcings  # ✅ SALVAR

        # Escaladores dinâmicos
        self.flow_scalers = {}
        self.climate_scalers = {}

        # Registrar escaladores
        self._register_scalers(train_indices)

        # Calcular eixos temporais
        (
            self.encoder_offsets,
            self.decoder_offsets,
            self.decoder_history,
            self.decoder_horizon,
        ) = compute_time_axes(flow_window_config, climate_window_config)

        self.encoder_length = len(self.encoder_offsets)
        self.decoder_length = len(self.decoder_offsets)

        # Construir índices válidos (com verificação de NaNs)
        self.valid_centers = self._build_valid_indices()

    def _get_cols_by_prefix(self, prefix: str, station: int) -> List[str]:
        """
        Retorna todas as colunas do DF que começam com 'prefix' e terminam com '_{station}'.
        Ordena alfabeticamente para garantir determinismo.
        """
        suffix = f"_{station}"
        cols = [
            c for c in self.df.columns
            if c.startswith(prefix) and c.endswith(suffix)
        ]
        cols.sort()
        return cols

    def _register_scalers(self, train_indices: np.ndarray):
        """Registra escaladores para todas as features."""
        def register_scaler(col_name):
            if col_name in self.df.columns:
                self.climate_scalers[col_name] = compute_scaler(
                    self.df[col_name].iloc[train_indices].to_numpy()
                )

        need_et = (self.forcings == "P_ET")

        for st in self.stations:
            # 1. Vazão Alvo
            q_col = f"Q_{st}"
            self.flow_scalers[q_col] = compute_scaler(
                self.df[q_col].iloc[train_indices].to_numpy()
            )

            # 2. Clima Básico
            # ✅ CORRIGIR: precipitation_chirps → precipitation_obs
            precip_obs = f"precipitation_obs_{st}"
            register_scaler(precip_obs)

            if need_et:
                et_obs = f"potential_evapotransp_gleam_{st}"
                register_scaler(et_obs)

            # ✅ UNPACKING CONDICIONAL
            forecast_info = self.forecast_cols[st]
            if isinstance(forecast_info, tuple):
                if len(forecast_info) == 2:
                    precip_fc, et_fc = forecast_info
                elif len(forecast_info) == 1:
                    precip_fc = forecast_info[0]
                    et_fc = None
                else:
                    raise ValueError(f"forecast_cols[{st}] deve ter 1 ou 2 elementos")
            else:
                # Se for string única
                precip_fc = forecast_info
                et_fc = None

            register_scaler(precip_fc)

            if need_et and et_fc is not None:
                register_scaler(et_fc)

            # ✅ ET forecast APENAS SE forcings="P_ET"
            if need_et:
                register_scaler(et_fc)

            # 3. Médias Móveis e Acumulados
            # ✅ CORRIGIR todos os prefixos
            for col in self._get_cols_by_prefix("precipitation_obs_ma", st):
                register_scaler(col)

            for col in self._get_cols_by_prefix("precipitation_forecast_ma", st):
                register_scaler(col)

            for col in self._get_cols_by_prefix("precipitation_obs_cum", st):
                register_scaler(col)

            for col in self._get_cols_by_prefix("precipitation_forecast_cum", st):
                register_scaler(col)

            # 4. APIs
            # ✅ CORRIGIR: api_chirps_k → api_obs_k
            for col in self._get_cols_by_prefix("api_obs_k", st):
                register_scaler(col)

            for col in self._get_cols_by_prefix("api_forecast_k", st):
                register_scaler(col)

            # 5. Derivadas e Regime
            register_scaler(f"dQ_dt_{st}")

            # ✅ CORRIGIR: dP_dt → dP_obs_dt
            register_scaler(f"dP_obs_dt_{st}")

            register_scaler(f"dP_dt_forecast_{st}")

            if f"regime_state_{st}" in self.df.columns:
                self.climate_scalers[f"regime_state_{st}"] = compute_scaler(
                    self.df[f"regime_state_{st}"].iloc[train_indices].astype(np.float32).to_numpy()
                )

            # 6. Log-Anomaly
            register_scaler(f"log_anomaly_{st}")
            for col in self._get_cols_by_prefix("log_anomaly_ma", st):
                register_scaler(col)

        # Escaladores estáticos
        self.static_scalers: Dict[str, Scaler] = {}
        for key in self.static_keys:
            vals = np.array([self.static_attrs[st][key] for st in self.stations], dtype=np.float32)
            self.static_scalers[key] = compute_scaler(vals)

    def _build_valid_indices(self) -> List[int]:
        """
        Constrói lista de índices válidos para centers.
        Agora considera reserva de dias ao final.
        """
        enc_min_offset = int(self.encoder_offsets[0])
        dec_max_offset = int(self.decoder_offsets[-1])

        window_size = dec_max_offset - enc_min_offset + 1
        flow_cols = [f"Q_{st}" for st in self.stations]

        valid_rows = self.df[flow_cols].notna().all(axis=1).astype(int)
        rolling_valid = valid_rows.rolling(window=window_size).min()
        rolling_valid_np = rolling_valid.to_numpy()

        start = -enc_min_offset

        # MUDANÇA CRÍTICA: Subtrair dias reservados
        end = len(self.df) - dec_max_offset - self.reserve_last_days

        valid_centers = []

        for c in range(start, end, self.window_stride):
            check_idx = c + dec_max_offset
            if 0 <= check_idx < len(rolling_valid_np):
                if rolling_valid_np[check_idx] == 1.0:
                    valid_centers.append(c)

        return valid_centers

    def __len__(self):
        return len(self.valid_centers)

    def _slice_series(self, series: pd.Series, center: int, offsets: np.ndarray) -> np.ndarray:
        """Corta uma série temporal com offsets relativos ao center."""
        idxs = center + offsets
        clip_mask = (idxs >= 0) & (idxs < len(series))
        values = np.full_like(offsets, fill_value=np.nan, dtype=np.float32)
        values[clip_mask] = series.iloc[idxs[clip_mask]].to_numpy()
        return values

    def _transform_and_append(self, col_name: str, center: int, offsets: np.ndarray, target_list: List[np.ndarray]):
        """Helper para cortar, escalar e adicionar à lista se a coluna existir."""
        if col_name in self.df.columns:
            series = self._slice_series(self.df[col_name], center, offsets)
            scaler = self.climate_scalers[col_name]
            norm_vals = scaler.transform(
                torch.from_numpy(series).float()
            ).numpy()
            target_list.append(norm_vals)

    def _build_flow_block(self, center: int, stage: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Constrói bloco de features de fluxo."""
        global_offsets = self.encoder_offsets if stage == "encoder" else self.decoder_offsets
        flow_arrays: List[np.ndarray] = []
        mask_arrays: List[np.ndarray] = []

        for station in self.stations:
            for spec in self.flow_window_config[station][stage]:
                spec_offsets = np.arange(spec["start"], spec["end"] + 1)
                raw_values = self._slice_series(
                    self.df[f"Q_{spec['source']}"],
                    center,
                    spec_offsets - spec["travel_time"],
                )

                aligned_values = np.full(len(global_offsets), np.nan, dtype=np.float32)
                aligned_mask = np.zeros(len(global_offsets), dtype=np.float32)

                for local_idx, spec_off in enumerate(spec_offsets):
                    match = np.where(global_offsets == spec_off)[0]
                    if match.size:
                        slot = match[0]
                        val = raw_values[local_idx]
                        if not np.isnan(val):
                            aligned_values[slot] = val
                            aligned_mask[slot] = 1.0

                scaler = self.flow_scalers[f"Q_{spec['source']}"]
                norm_values = scaler.transform(
                    torch.from_numpy(np.nan_to_num(aligned_values, nan=scaler.mean)).float()
                ).numpy()

                flow_arrays.append(norm_values)
                mask_arrays.append(aligned_mask)

        return flow_arrays, mask_arrays

    def _build_climate_block(self, center: int, stage: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Constrói bloco de features climáticas."""
        offsets = self.encoder_offsets if stage == "encoder" else self.decoder_offsets
        obs_arrays: List[np.ndarray] = []
        fc_arrays: List[np.ndarray] = []

        for station in self.stations:
            if stage == "encoder":
                self._add_encoder_features(station, center, offsets, obs_arrays)
            else:
                self._add_decoder_features(station, center, offsets, fc_arrays)

        return obs_arrays, fc_arrays

    def _add_encoder_features(self, station: int, center: int, offsets: np.ndarray, arrays: List[np.ndarray]):
        """Adiciona features do encoder de forma dinâmica."""
        need_et = (self.forcings == "P_ET")

        # 1. Clima Básico
        # ✅ CORRIGIR: precipitation_chirps → precipitation_obs
        self._transform_and_append(f"precipitation_obs_{station}", center, offsets, arrays)

        if need_et:
            # ✅ CORRIGIR: potential_evapotransp_gleam é o nome correto
            self._transform_and_append(f"potential_evapotransp_gleam_{station}", center, offsets, arrays)

        # 2. Médias Móveis e Acumulados
        # ✅ CORRIGIR: precipitation_chirps_ma → precipitation_obs_ma
        for col in self._get_cols_by_prefix("precipitation_obs_ma", station):
            self._transform_and_append(col, center, offsets, arrays)

        # ✅ CORRIGIR: precipitation_chirps_cum → precipitation_obs_cum
        for col in self._get_cols_by_prefix("precipitation_obs_cum", station):
            self._transform_and_append(col, center, offsets, arrays)

        # 3. APIs
        # ✅ CORRIGIR: api_chirps_k → api_obs_k
        for col in self._get_cols_by_prefix("api_obs_k", station):
            self._transform_and_append(col, center, offsets, arrays)

        # 4. Derivadas e Regime
        self._transform_and_append(f"dQ_dt_{station}", center, offsets, arrays)

        # ✅ CORRIGIR: dP_dt → dP_obs_dt
        self._transform_and_append(f"dP_obs_dt_{station}", center, offsets, arrays)

        self._transform_and_append(f"regime_state_{station}", center, offsets, arrays)

        # 5. Log Anomaly
        self._transform_and_append(f"log_anomaly_{station}", center, offsets, arrays)
        for col in self._get_cols_by_prefix("log_anomaly_ma", station):
            self._transform_and_append(col, center, offsets, arrays)

    def _add_decoder_features(self, station: int, center: int, offsets: np.ndarray, arrays: List[np.ndarray]):
        """Adiciona features do decoder de forma dinâmica."""
        need_et = (self.forcings == "P_ET")

        # ✅ UNPACKING CONDICIONAL
        forecast_info = self.forecast_cols[station]
        if isinstance(forecast_info, tuple):
            if len(forecast_info) == 2:
                fc_p_name, fc_e_name = forecast_info
            elif len(forecast_info) == 1:
                fc_p_name = forecast_info[0]
                fc_e_name = None
            else:
                raise ValueError(f"forecast_cols[{station}] deve ter 1 ou 2 elementos")
        else:
            fc_p_name = forecast_info
            fc_e_name = None

        self._transform_and_append(fc_p_name, center, offsets, arrays)

        if need_et and fc_e_name is not None:
            self._transform_and_append(fc_e_name, center, offsets, arrays)

        # Médias e Acumulados de Forecast
        for col in self._get_cols_by_prefix("precipitation_forecast_ma", station):
            self._transform_and_append(col, center, offsets, arrays)

        for col in self._get_cols_by_prefix("precipitation_forecast_cum", station):
            self._transform_and_append(col, center, offsets, arrays)

        # APIs de Forecast
        for col in self._get_cols_by_prefix("api_forecast_k", station):
            self._transform_and_append(col, center, offsets, arrays)

        self._transform_and_append(f"dP_dt_forecast_{station}", center, offsets, arrays)

    def _build_temporal_block(self, center: int, stage: str) -> np.ndarray:
        """Constrói bloco de features temporais."""
        offsets = self.encoder_offsets if stage == "encoder" else self.decoder_offsets
        idxs = center + offsets
        temp_features = []
        for feature in self.temporal_features:
            series = self.df[feature].to_numpy()
            values = np.zeros(len(offsets), dtype=np.float32)
            mask = (idxs >= 0) & (idxs < len(series))
            values[mask] = series[idxs[mask]]
            temp_features.append(values)
        return np.stack(temp_features, axis=-1)

    def _build_static_vector(self, center: int) -> np.ndarray:
        """Constrói vetor de features estáticas."""
        static_feats: List[float] = []
        for station in self.stations:
            attrs = self.static_attrs[station]
            for k in self.static_keys:
                scaler = self.static_scalers[k]
                val = torch.tensor(attrs[k], dtype=torch.float32)
                static_feats.append(scaler.transform(val).item())
        return np.array(static_feats, dtype=np.float32)

    def _build_target(self, center: int) -> np.ndarray:
        """Constrói alvo para previsão."""
        future_offsets = self.decoder_offsets[self.decoder_offsets >= 0]
        target = np.zeros((len(future_offsets), len(self.stations)), dtype=np.float32)
        for st_idx, station in enumerate(self.stations):
            scaler = self.flow_scalers[f"Q_{station}"]
            series = self.df[f"Q_{station}"].to_numpy()
            idxs = center + future_offsets
            values = np.zeros_like(future_offsets, dtype=np.float32)
            mask = (idxs >= 0) & (idxs < len(series))
            values[mask] = series[idxs[mask]]
            target[:, st_idx] = scaler.transform(torch.from_numpy(values).float()).numpy()
        return target

    def __getitem__(self, idx):
        center = self.valid_centers[idx]

        flow_enc, mask_enc = self._build_flow_block(center, "encoder")
        flow_dec, mask_dec = self._build_flow_block(center, "decoder")

        climate_obs_enc, climate_fc_enc = self._build_climate_block(center, "encoder")
        climate_obs_dec, climate_fc_dec = self._build_climate_block(center, "decoder")

        temp_enc = self._build_temporal_block(center, "encoder")
        temp_dec = self._build_temporal_block(center, "decoder")
        static_vec = self._build_static_vector(center)
        target = self._build_target(center)

        encoder_dyn = np.stack(flow_enc + climate_obs_enc, axis=-1)
        decoder_dyn = np.stack(flow_dec + climate_fc_dec, axis=-1)

        mask_enc = np.stack(mask_enc, axis=-1)
        mask_dec = np.stack(mask_dec, axis=-1)

        last_obs = []
        for st in self.stations:
            scaler = self.flow_scalers[f"Q_{st}"]
            last_value = self.df[f"Q_{st}"].iloc[center - 1]
            last_obs.append(scaler.transform(torch.tensor(last_value).float()).item())
        last_obs = torch.tensor(last_obs, dtype=torch.float32)

        forecast_start_idx = center + self.decoder_offsets[self.decoder_history]
        if 0 <= forecast_start_idx < len(self.df):
            forecast_date = self.df.index[forecast_start_idx]
        else:
            forecast_date = None

        return Sample(
            encoder_dyn=torch.tensor(encoder_dyn, dtype=torch.float32),
            decoder_dyn=torch.tensor(decoder_dyn, dtype=torch.float32),
            static=torch.tensor(static_vec, dtype=torch.float32),
            temporal_enc=torch.tensor(temp_enc, dtype=torch.float32),
            temporal_dec=torch.tensor(temp_dec, dtype=torch.float32),
            target=torch.tensor(target, dtype=torch.float32),
            mask_enc=torch.tensor(mask_enc, dtype=torch.float32),
            mask_dec=torch.tensor(mask_dec, dtype=torch.float32),
            baseline_last=last_obs,
            forecast_date=forecast_date,
            date_index=center,
        )


def create_temporal_split_with_gap(
    dataset: HydroDataset,
    train_ratio: float = 0.95,
    gap: Optional[int] = None,
    exclude_reserved: bool = True
) -> Tuple[Subset, Subset]:
    """
    Cria split temporal com gap.

    Args:
        exclude_reserved: Se True, não inclui dias reservados no split
    """
    n = len(dataset)  # Já considera reserve_last_days

    # O dataset já removeu os índices reservados, então n já é "correto"
    split_idx = int(n * train_ratio)
    split_idx = min(max(split_idx, 1), n - 1)

    enc_min = int(dataset.encoder_offsets[0])
    dec_max = int(dataset.decoder_offsets[-1])
    min_gap_centers = dec_max - enc_min + 1

    eff_gap = min_gap_centers if gap is None else max(gap, min_gap_centers)
    val_start = min(split_idx + eff_gap, n)

    train_indices = list(range(0, split_idx))
    val_indices = list(range(val_start, n))

    if len(val_indices) == 0:
        val_indices = list(range(split_idx, n))

    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def create_dataset_for_training_validation(
    df: pd.DataFrame,
    stations: List[int],
    static_attrs: Dict[int, Dict[str, float]],
    train_indices: np.ndarray,
    forecast_cols: Dict[int, Tuple[str, str]],
    flow_window_config: Dict,
    climate_window_config: Dict,
    temporal_features: List[str],
    api_k_list: List[float],
    static_keys: List[str],
    horizon: int = 18,
    use_last_days_as_forecast: bool = True,
    window_stride: int = 1,
    forcings: str = "P"
) -> HydroDataset:
    """
    Cria dataset apropriado para treino ou produção.

    Args:
        df: DataFrame com features processadas
        stations: Lista de IDs das estações
        static_attrs: Dicionário de atributos estáticos
        train_indices: Índices do conjunto de treino para calcular scalers
        forecast_cols: Mapeamento de colunas de forecast por estação
        flow_window_config: Configuração de janelas de vazão
        climate_window_config: Configuração de janelas climáticas
        temporal_features: Lista de features temporais
        api_k_list: Lista de valores k para API
        static_keys: Chaves dos atributos estáticos
        horizon: Horizonte de previsão em dias
        use_last_days_as_forecast: Se True, reserva últimos 'horizon' dias
                                    para simular forecast durante treino
        window_stride: Passo entre janelas deslizantes

    Returns:
        HydroDataset configurado

    Examples:
        # Para treino (reservar últimos 18 dias como "forecast")
         dataset_train = create_dataset_for_training_validation(
             df=combined_df,
             stations=[10100000, 13150000, 14100000],
             #  outros parâmetros
             horizon=18,
             use_last_days_as_forecast=True
         )

        # Para produção (usar todos os dados)
         dataset_prod = create_dataset_for_training_validation(
             df=combined_df,
             stations=[10100000, 13150000, 14100000],
             #  outros parâmetros
             horizon=18,
             use_last_days_as_forecast=False
         )
    """
    reserve_days = horizon if use_last_days_as_forecast else 0

    return HydroDataset(
        df=df,
        stations=stations,
        static_attrs=static_attrs,
        train_indices=train_indices,
        forecast_cols=forecast_cols,
        flow_window_config=flow_window_config,
        climate_window_config=climate_window_config,
        temporal_features=temporal_features,
        api_k_list=api_k_list,
        static_keys=static_keys,
        window_stride=window_stride,
        reserve_last_days=reserve_days,
        forcings=forcings
    )

def create_dataset_for_inference(
    df: pd.DataFrame,
    stations: List[int],
    static_attrs: Dict[int, Dict[str, float]],
    forecast_cols: Dict[int, Tuple[str, str]],
    flow_window_config: Dict,
    climate_window_config: Dict,
    temporal_features: List[str],
    api_k_list: List[float],
    static_keys: List[str],
    reference_dates: Optional[List[pd.Timestamp]] = None,
    forcings: str = "P"
) -> HydroDataset:
    """
    Cria dataset para inferência em produção.

    Diferença para `create_dataset_for_training`:
    - Não faz sliding window em todos os dados
    - Cria janelas APENAS para datas de referência específicas
    - Usa TODOS os dados disponíveis (sem reserva)
    - Se reference_dates=None, usa apenas a última data válida

    Args:
        df: DataFrame com features processadas
        stations: Lista de IDs das estações
        static_attrs: Dicionário de atributos estáticos
        forecast_cols: Mapeamento de colunas de forecast por estação
        flow_window_config: Configuração de janelas de vazão
        climate_window_config: Configuração de janelas climáticas
        temporal_features: Lista de features temporais
        api_k_list: Lista de valores k para API
        static_keys: Chaves dos atributos estáticos
        reference_dates: Lista com o ÚLTIMO DIA OBSERVADO para cada janela de inferência.
                        Aceita strings ('2026-03-16'), Timestamps ou lista mista.
                        Se None, usa apenas a última data válida do dataset.

    Returns:
        HydroDataset configurado para inferência

    Examples:
        # Inferência para a última data disponível
         ds_inference = create_dataset_for_inference(
             df=combined_df,
             stations=[10100000, 13150000, 14100000],
             #  outros parâmetros
         )
         print(len(ds_inference))  # 1 janela

        # Inferência para múltiplas datas específicas (último dia observado de cada janela)
         ds_inference = create_dataset_for_inference(
             df=combined_df,
             #  outros parâmetros
             reference_dates=['2026-03-16', '2026-03-17']
         )
         print(len(ds_inference))  # 2 janelas
    """
    # Criar dataset base SEM sliding window
    # Usamos train_indices dummy (primeiro índice válido)
    train_indices = np.array([0])

    dataset = HydroDataset(
        df=df,
        stations=stations,
        static_attrs=static_attrs,
        train_indices=train_indices,
        forecast_cols=forecast_cols,
        flow_window_config=flow_window_config,
        climate_window_config=climate_window_config,
        temporal_features=temporal_features,
        api_k_list=api_k_list,
        static_keys=static_keys,
        window_stride=1,
        reserve_last_days=0,
        forcings=forcings
    )

    # Se não especificou datas, usar apenas a última válida
    if reference_dates is None:
        if len(dataset.valid_centers) > 0:
            last_center = dataset.valid_centers[-1]
            dataset.valid_centers = [last_center]

            forecast_start_idx = last_center + dataset.decoder_offsets[dataset.decoder_history]
            if 0 <= forecast_start_idx < len(df):
                forecast_date = df.index[forecast_start_idx]
                print(f"📅 Inferência configurada para data de referência: {forecast_date.date()}")
                print("(Último dado observado disponível para encoder)")
        else:
            raise ValueError("Nenhum center válido encontrado no dataset")

    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame deve ter índice DatetimeIndex para usar reference_dates")

        # Garantir que reference_dates são pd.Timestamp (aceita strings, Timestamps, etc.)
        reference_dates = pd.to_datetime(reference_dates)

        enc_min_offset = int(dataset.encoder_offsets[0])    # normalmente -30
        enc_max_offset = int(dataset.encoder_offsets[-1])   # normalmente -1
        dec_max_offset = int(dataset.decoder_offsets[-1])   # normalmente +14

        custom_centers = []

        for ref_date in reference_dates:
            # Encontrar índice da data no DataFrame
            try:
                date_idx = df.index.get_loc(ref_date)
            except KeyError:
                print(f"⚠️  Data {ref_date.date()} não encontrada no DataFrame, pulando")
                continue

            # ref_date é o ÚLTIMO DIA OBSERVADO = posição (center + enc_max_offset)
            # logo: center = date_idx - enc_max_offset
            # ex: enc_max_offset = -1  →  center = date_idx + 1  (primeiro dia do forecast)
            center = date_idx - enc_max_offset

            # Para inferência, só exigimos que o trecho do ENCODER esteja dentro dos bounds.
            # O trecho do decoder pode ultrapassar o fim do df (dados futuros com NaN é esperado).
            if center + enc_min_offset >= 0 and center + enc_max_offset < len(df):
                # Verificar NaN apenas no trecho do encoder (passado observado).
                # O trecho do decoder terá NaN por design (dados futuros não existem).
                flow_cols = [f"Q_{st}" for st in stations]
                encoder_slice = df.iloc[center + enc_min_offset:center + enc_max_offset + 1][flow_cols]

                if not encoder_slice.isna().any().any():
                    custom_centers.append(center)
                    print(f"✅ Data {ref_date.date()} adicionada (center={center}, forecast a partir de {df.index[center].date()})")
                else:
                    print(f"⚠️  Data {ref_date.date()} tem NaN na janela do encoder, pulando")
            else:
                print(f"⚠️  Data {ref_date.date()} não tem dados suficientes no encoder, pulando")
                print(f"   (center={center}, enc_min={enc_min_offset}, enc_max={enc_max_offset}, len(df)={len(df)})")

        if len(custom_centers) == 0:
            raise ValueError("Nenhuma data válida encontrada para inferência")

        dataset.valid_centers = custom_centers
        print(f"\n📊 Dataset de inferência criado com {len(custom_centers)} janela(s)")

    return dataset
