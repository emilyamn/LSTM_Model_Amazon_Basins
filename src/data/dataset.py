"""
Dataset hidrológico para treino de modelos de deep learning.

- Série de precipitação UNIFICADA: `precipitation_{station}` para encoder e decoder.
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
        flow_window_config: Dict,
        climate_window_config: Dict,
        temporal_features: List[str],
        api_k_list: List[float],
        static_keys: List[str],
        window_stride: int = 1,
        reserve_last_days: int = 0,
        forcings: str = "P",
        # Parâmetro legado — aceito mas ignorado
        forecast_cols: Optional[Dict] = None,
    ):
        if forecast_cols is not None:
            print(
                "⚠️  'forecast_cols' não é mais utilizado (série unificada). "
                "Parâmetro ignorado."
            )
        self.df = df
        self.stations = stations
        self.static_attrs = static_attrs
        self.flow_window_config = flow_window_config
        self.climate_window_config = climate_window_config
        self.temporal_features = temporal_features
        self.api_k_list = api_k_list
        self.static_keys = static_keys
        self.window_stride = window_stride
        self.reserve_last_days = reserve_last_days
        self.forcings = forcings

        self.flow_scalers: Dict[str, Scaler] = {}
        self.climate_scalers: Dict[str, Scaler] = {}

        self._register_scalers(train_indices)

        (
            self.encoder_offsets,
            self.decoder_offsets,
            self.decoder_history,
            self.decoder_horizon,
        ) = compute_time_axes(flow_window_config, climate_window_config)

        self.encoder_length = len(self.encoder_offsets)
        self.decoder_length = len(self.decoder_offsets)
        self.valid_centers = self._build_valid_indices()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_cols_by_prefix(self, prefix: str, station: int) -> List[str]:
        """Retorna colunas que começam com `prefix` e terminam com `_{station}`, ordenadas."""
        suffix = f"_{station}"
        return sorted(
            c for c in self.df.columns
            if c.startswith(prefix) and c.endswith(suffix)
        )

    # ------------------------------------------------------------------
    # Scalers
    # ------------------------------------------------------------------

    def _register_scalers(self, train_indices: np.ndarray):
        """Registra escaladores para todas as features usando os novos nomes de colunas."""
        silent = len(train_indices) <= 1  # dummy indices → suprimir warnings

        def register(col_name: str):
            if col_name in self.df.columns:
                self.climate_scalers[col_name] = compute_scaler(
                    self.df[col_name].iloc[train_indices].to_numpy(), silent=silent
                )

        need_et = self.forcings == "P_ET"

        for st in self.stations:
            # Vazão
            self.flow_scalers[f"Q_{st}"] = compute_scaler(
                self.df[f"Q_{st}"].iloc[train_indices].to_numpy(), silent=silent
            )

            # Precipitação unificada
            register(f"precipitation_{st}")

            # ET unificada
            if need_et:
                register(f"et_{st}")

            # Médias móveis e acumulados (prefixo unificado)
            for col in self._get_cols_by_prefix("precipitation_ma", st):
                register(col)
            for col in self._get_cols_by_prefix("precipitation_cum", st):
                register(col)

            # ET médias móveis
            if need_et:
                for col in self._get_cols_by_prefix("et_ma", st):
                    register(col)

            # API (prefixo unificado: api_k)
            for col in self._get_cols_by_prefix("api_k", st):
                register(col)

            # Derivadas e regime
            register(f"dQ_dt_{st}")
            register(f"dP_dt_{st}")

            if f"regime_state_{st}" in self.df.columns:
                self.climate_scalers[f"regime_state_{st}"] = compute_scaler(
                    self.df[f"regime_state_{st}"].iloc[train_indices].astype(np.float32).to_numpy(),
                    silent=silent
                )

            # Log-anomaly
            register(f"log_anomaly_{st}")
            for col in self._get_cols_by_prefix("log_anomaly_ma", st):
                register(col)

        # Escaladores estáticos
        self.static_scalers: Dict[str, Scaler] = {}
        for key in self.static_keys:
            vals = np.array(
                [self.static_attrs[st][key] for st in self.stations], dtype=np.float32
            )
            self.static_scalers[key] = compute_scaler(vals, silent=silent)

    # ------------------------------------------------------------------
    # Índices válidos
    # ------------------------------------------------------------------

    def _build_valid_indices(self) -> List[int]:
        enc_min_offset = int(self.encoder_offsets[0])
        dec_max_offset = int(self.decoder_offsets[-1])
        window_size = dec_max_offset - enc_min_offset + 1

        flow_cols = [f"Q_{st}" for st in self.stations]
        valid_rows = self.df[flow_cols].notna().all(axis=1).astype(int)
        rolling_valid = valid_rows.rolling(window=window_size).min().to_numpy()

        start = -enc_min_offset
        end = len(self.df) - dec_max_offset - self.reserve_last_days

        return [
            c for c in range(start, end, self.window_stride)
            if 0 <= c + dec_max_offset < len(rolling_valid)
            and rolling_valid[c + dec_max_offset] == 1.0
        ]

    def __len__(self):
        return len(self.valid_centers)

    # ------------------------------------------------------------------
    # Construção de blocos
    # ------------------------------------------------------------------

    def _slice_series(
        self, series: pd.Series, center: int, offsets: np.ndarray
    ) -> np.ndarray:
        idxs = center + offsets
        clip = (idxs >= 0) & (idxs < len(series))
        values = np.full(len(offsets), np.nan, dtype=np.float32)
        values[clip] = series.iloc[idxs[clip]].to_numpy()
        return values

    def _transform_and_append(
        self,
        col_name: str,
        center: int,
        offsets: np.ndarray,
        target_list: List[np.ndarray],
    ):
        """Corta, escala e adiciona à lista se a coluna existir."""
        if col_name in self.df.columns and col_name in self.climate_scalers:
            series = self._slice_series(self.df[col_name], center, offsets)
            scaler = self.climate_scalers[col_name]
            target_list.append(
                scaler.transform(torch.from_numpy(series).float()).numpy()
            )

    def _build_flow_block(
        self, center: int, stage: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        global_offsets = self.encoder_offsets if stage == "encoder" else self.decoder_offsets
        flow_arrays, mask_arrays = [], []
        station_indices: List[int] = []

        for st_idx, station in enumerate(self.stations):
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
                    torch.from_numpy(
                        np.nan_to_num(aligned_values, nan=scaler.mean)
                    ).float()
                ).numpy()

                flow_arrays.append(norm_values)
                mask_arrays.append(aligned_mask)
                station_indices.append(st_idx)

        return flow_arrays, mask_arrays, station_indices

    def _build_climate_block(
        self, center: int, stage: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        offsets = self.encoder_offsets if stage == "encoder" else self.decoder_offsets
        arrays: List[np.ndarray] = []

        for station in self.stations:
            if stage == "encoder":
                self._add_encoder_features(station, center, offsets, arrays)
            else:
                self._add_decoder_features(station, center, offsets, arrays)

        # Retorna (obs_arrays, fc_arrays) para manter compatibilidade com __getitem__
        # Agora ambos são o mesmo — retornamos arrays em obs e lista vazia em fc
        return arrays, []

    def _add_encoder_features(
        self,
        station: int,
        center: int,
        offsets: np.ndarray,
        arrays: List[np.ndarray],
    ):
        """Features do encoder: precipitação unificada + derivadas de fluxo."""
        need_et = self.forcings == "P_ET"

        # Precipitação unificada
        self._transform_and_append(f"precipitation_{station}", center, offsets, arrays)

        # ET
        if need_et:
            self._transform_and_append(f"et_{station}", center, offsets, arrays)

        # Médias móveis e acumulados
        for col in self._get_cols_by_prefix("precipitation_ma", station):
            self._transform_and_append(col, center, offsets, arrays)
        for col in self._get_cols_by_prefix("precipitation_cum", station):
            self._transform_and_append(col, center, offsets, arrays)

        # ET médias móveis
        if need_et:
            for col in self._get_cols_by_prefix("et_ma", station):
                self._transform_and_append(col, center, offsets, arrays)

        # API
        for col in self._get_cols_by_prefix("api_k", station):
            self._transform_and_append(col, center, offsets, arrays)

        # Derivadas de fluxo e precipitação (disponíveis no passado)
        self._transform_and_append(f"dQ_dt_{station}", center, offsets, arrays)
        self._transform_and_append(f"dP_dt_{station}", center, offsets, arrays)
        self._transform_and_append(f"regime_state_{station}", center, offsets, arrays)

        # Log-anomaly
        self._transform_and_append(f"log_anomaly_{station}", center, offsets, arrays)
        for col in self._get_cols_by_prefix("log_anomaly_ma", station):
            self._transform_and_append(col, center, offsets, arrays)

    def _add_decoder_features(
        self,
        station: int,
        center: int,
        offsets: np.ndarray,
        arrays: List[np.ndarray],
    ):
        """
        Features do decoder: precipitação unificada (parte histórica = obs, horizonte = forecast).
        Sem derivadas de fluxo (Q desconhecido no futuro).
        """
        need_et = self.forcings == "P_ET"

        # Precipitação unificada
        self._transform_and_append(f"precipitation_{station}", center, offsets, arrays)

        # ET
        if need_et:
            self._transform_and_append(f"et_{station}", center, offsets, arrays)

        # Médias móveis e acumulados
        for col in self._get_cols_by_prefix("precipitation_ma", station):
            self._transform_and_append(col, center, offsets, arrays)
        for col in self._get_cols_by_prefix("precipitation_cum", station):
            self._transform_and_append(col, center, offsets, arrays)

        # ET médias móveis
        if need_et:
            for col in self._get_cols_by_prefix("et_ma", station):
                self._transform_and_append(col, center, offsets, arrays)

        # API
        for col in self._get_cols_by_prefix("api_k", station):
            self._transform_and_append(col, center, offsets, arrays)

        # dP_dt disponível (série unificada inclui forecast)
        self._transform_and_append(f"dP_dt_{station}", center, offsets, arrays)

        # NÃO incluir: dQ_dt, regime_state, log_anomaly (fluxo desconhecido no futuro)

    def _build_temporal_block(self, center: int, stage: str) -> np.ndarray:
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
        static_feats = []
        for station in self.stations:
            attrs = self.static_attrs[station]
            for k in self.static_keys:
                scaler = self.static_scalers[k]
                val = torch.tensor(attrs[k], dtype=torch.float32)
                static_feats.append(scaler.transform(val).item())
        return np.array(static_feats, dtype=np.float32)

    def _build_target(self, center: int) -> Tuple[np.ndarray, np.ndarray]:
        """Constrói alvo e máscara de validade para o horizonte de previsão."""
        future_offsets = self.decoder_offsets[self.decoder_offsets >= 0]
        target = np.zeros((len(future_offsets), len(self.stations)), dtype=np.float32)
        target_mask = np.zeros((len(future_offsets), len(self.stations)), dtype=np.float32)

        for st_idx, station in enumerate(self.stations):
            scaler = self.flow_scalers[f"Q_{station}"]
            series = self.df[f"Q_{station}"].to_numpy()
            idxs = center + future_offsets
            for i, idx in enumerate(idxs):
                if 0 <= idx < len(series) and not np.isnan(series[idx]):
                    target[i, st_idx] = scaler.transform(
                        torch.tensor(series[idx]).float()
                    ).item()
                    target_mask[i, st_idx] = 1.0

        return target, target_mask

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        center = self.valid_centers[idx]

        flow_enc, mask_enc, _ = self._build_flow_block(center, "encoder")
        flow_dec, mask_dec, dec_station_indices = self._build_flow_block(center, "decoder")

        climate_enc, _ = self._build_climate_block(center, "encoder")
        climate_dec, _ = self._build_climate_block(center, "decoder")

        temp_enc = self._build_temporal_block(center, "encoder")
        temp_dec = self._build_temporal_block(center, "decoder")
        static_vec = self._build_static_vector(center)
        target, target_mask = self._build_target(center)

        encoder_dyn = np.stack(flow_enc + climate_enc, axis=-1)
        decoder_dyn = np.stack(flow_dec + climate_dec, axis=-1)

        mask_enc_arr = np.stack(mask_enc, axis=-1)
        mask_dec_history_full = np.stack(mask_dec, axis=-1)  # (decoder_length, n_flow_cols)

        # Reduzir mask de n_flow_cols → n_stations usando mapeamento real
        # Para cada estação: mask = min das masks de todos os seus specs
        n_stations = len(self.stations)
        mask_dec_history = np.ones(
            (self.decoder_history, n_stations), dtype=np.float32
        )
        for col_idx, st_idx in enumerate(dec_station_indices):
            mask_dec_history[:, st_idx] = np.minimum(
                mask_dec_history[:, st_idx],
                mask_dec_history_full[:self.decoder_history, col_idx],
            )

        full_mask_dec = np.concatenate([
            mask_dec_history,   # (decoder_history, n_stations)
            target_mask,        # (decoder_horizon, n_stations)
        ], axis=0)

        # baseline_last: último valor observado de cada estação (antes do horizonte)
        last_obs = []
        for st in self.stations:
            scaler = self.flow_scalers[f"Q_{st}"]
            last_value = self.df[f"Q_{st}"].iloc[center - 1]
            last_obs.append(scaler.transform(torch.tensor(last_value).float()).item())
        last_obs = torch.tensor(last_obs, dtype=torch.float32)

        forecast_start_idx = center + self.decoder_offsets[self.decoder_history]
        forecast_date = (
            self.df.index[forecast_start_idx]
            if 0 <= forecast_start_idx < len(self.df)
            else None
        )

        return Sample(
            encoder_dyn=torch.tensor(encoder_dyn, dtype=torch.float32),
            decoder_dyn=torch.tensor(decoder_dyn, dtype=torch.float32),
            static=torch.tensor(static_vec, dtype=torch.float32),
            temporal_enc=torch.tensor(temp_enc, dtype=torch.float32),
            temporal_dec=torch.tensor(temp_dec, dtype=torch.float32),
            target=torch.tensor(target, dtype=torch.float32),
            mask_enc=torch.tensor(mask_enc_arr, dtype=torch.float32),
            mask_dec=torch.tensor(full_mask_dec, dtype=torch.float32),
            baseline_last=last_obs,
            forecast_date=forecast_date,
            date_index=center,
        )


# ==============================================================================
# Funções de criação de dataset
# ==============================================================================

def create_temporal_split_with_gap(
    dataset: HydroDataset,
    train_ratio: float = 0.95,
    val_ratio: float = 0.025,
    test_ratio: float = 0.025,
    gap: Optional[int] = None,
    exclude_reserved: bool = True,
) -> Tuple[Subset, Subset, Subset]:
    """Cria split temporal com gap entre treino e validação."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, rtol=1e-5):
        raise ValueError(
            f"Proporções devem somar 1.0. "
            f"train={train_ratio}, val={val_ratio}, test={test_ratio} → {total_ratio}"
        )

    n = len(dataset)
    train_end_idx = min(max(int(n * train_ratio), 1), n - 1)
    val_start_idx = min(max(int(n * (train_ratio + val_ratio)), train_end_idx + 1), n)

    enc_min = int(dataset.encoder_offsets[0])
    dec_max = int(dataset.decoder_offsets[-1])
    min_gap = dec_max - enc_min + 1
    eff_gap = min_gap if gap is None else max(gap, min_gap)

    val_start_with_gap = min(train_end_idx + eff_gap, n)
    train_indices = list(range(0, train_end_idx))

    val_indices = (
        list(range(val_start_with_gap, val_start_idx))
        if val_start_with_gap < val_start_idx
        else list(range(train_end_idx, val_start_idx))
    )
    test_indices = list(range(val_start_idx, n))

    if len(val_indices) == 0:
        if len(test_indices) > 1:
            split = len(test_indices) // 2
            val_indices = test_indices[:split]
            test_indices = test_indices[split:]
        else:
            val_indices = list(range(train_end_idx, n))
            test_indices = []

    if len(test_indices) == 0:
        test_indices = [n - 1] if n > 1 else []

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def create_dataset_for_training_validation(
    df: pd.DataFrame,
    stations: List[int],
    static_attrs: Dict[int, Dict[str, float]],
    train_indices: np.ndarray,
    flow_window_config: Dict,
    climate_window_config: Dict,
    temporal_features: List[str],
    api_k_list: List[float],
    static_keys: List[str],
    horizon: int = 18,
    use_last_days_as_forecast: bool = True,
    window_stride: int = 1,
    forcings: str = "P",
    # Parâmetro legado
    forecast_cols: Optional[Dict] = None,
) -> HydroDataset:
    """Cria dataset para treino/validação/teste."""
    return HydroDataset(
        df=df,
        stations=stations,
        static_attrs=static_attrs,
        train_indices=train_indices,
        flow_window_config=flow_window_config,
        climate_window_config=climate_window_config,
        temporal_features=temporal_features,
        api_k_list=api_k_list,
        static_keys=static_keys,
        window_stride=window_stride,
        reserve_last_days=horizon if use_last_days_as_forecast else 0,
        forcings=forcings,
        forecast_cols=forecast_cols,  # ignorado internamente
    )


def create_dataset_for_inference(
    df: pd.DataFrame,
    stations: List[int],
    static_attrs: Dict[int, Dict[str, float]],
    flow_window_config: Dict,
    climate_window_config: Dict,
    temporal_features: List[str],
    api_k_list: List[float],
    static_keys: List[str],
    meta: dict,
    reference_dates: Optional[List[pd.Timestamp]] = None,
    forcings: str = "P",
    # Parâmetro legado
    forecast_cols: Optional[Dict] = None,
) -> HydroDataset:
    """
    Cria dataset para inferência operacional.

    Args:
        meta: Dicionário com scalers do treino (obrigatório).
              Deve conter: 'flow_scalers', 'climate_scalers', 'static_scalers'.
    """
    dataset = HydroDataset(
        df=df,
        stations=stations,
        static_attrs=static_attrs,
        train_indices=np.array([0]),  # dummy — será substituído pelos scalers do meta
        flow_window_config=flow_window_config,
        climate_window_config=climate_window_config,
        temporal_features=temporal_features,
        api_k_list=api_k_list,
        static_keys=static_keys,
        window_stride=1,
        reserve_last_days=0,
        forcings=forcings,
        forecast_cols=forecast_cols,  # ignorado
    )

    # Injetar scalers do treino (sobrescreve os dummy)
    dataset.flow_scalers    = meta["flow_scalers"]
    dataset.climate_scalers = meta["climate_scalers"]
    dataset.static_scalers  = meta["static_scalers"]

    # Configurar janelas válidas
    if reference_dates is None:
        if len(dataset.valid_centers) == 0:
            raise ValueError("Nenhum center válido encontrado no dataset.")
        last_center = dataset.valid_centers[-1]
        dataset.valid_centers = [last_center]
        forecast_start_idx = last_center + dataset.decoder_offsets[dataset.decoder_history]
        if 0 <= forecast_start_idx < len(df):
            print(f"📅 Inferência: {df.index[forecast_start_idx].date()}")
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame deve ter índice DatetimeIndex.")

        reference_dates = pd.to_datetime(reference_dates)
        enc_min_offset = int(dataset.encoder_offsets[0])
        enc_max_offset = int(dataset.encoder_offsets[-1])
        flow_cols = [f"Q_{st}" for st in stations]
        custom_centers = []

        for ref_date in reference_dates:
            try:
                date_idx = df.index.get_loc(ref_date)
            except KeyError:
                print(f"⚠️  Data {ref_date.date()} não encontrada, pulando.")
                continue

            center = date_idx - enc_max_offset

            if center + enc_min_offset >= 0 and center + enc_max_offset < len(df):
                enc_slice = df.iloc[
                    center + enc_min_offset: center + enc_max_offset + 1
                ][flow_cols]
                if not enc_slice.isna().any().any():
                    custom_centers.append(center)
                    print(
                        f"✅ {ref_date.date()} adicionada "
                        f"(forecast a partir de {df.index[center].date()})"
                    )
                else:
                    print(f"⚠️  {ref_date.date()}: NaN no encoder, pulando.")
            else:
                print(f"⚠️  {ref_date.date()}: dados insuficientes no encoder, pulando.")

        if len(custom_centers) == 0:
            raise ValueError("Nenhuma data válida para inferência.")

        dataset.valid_centers = custom_centers
        print(f"\n📊 Dataset de inferência: {len(custom_centers)} janela(s)")

    return dataset
