"""
Módulo para feature engineering de dados hidrológicos.

Mudanças principais (v2):
- Série de precipitação UNIFICADA: uma coluna `precipitation_{station}` contínua
  (obs até d0, forecast após d0). Elimina a duplicação _obs / _forecast.
- Todas as features derivadas (ma, cum, api, dP_dt) calculadas sobre a série unificada.
- ET também unificada em `et_{station}` quando forcings="P_ET".
- column_names aceita lista [flow, precip] ou [flow, precip, et] além de dict.
- Parâmetros forecast_ma_windows / forecast_cumulative_windows removidos
  (eram idênticos aos de obs durante treino).
"""

from typing import Dict, List, Optional, Literal, Union
import pathlib
import re
import sys
import pandas as pd
import numpy as np

current_dir = pathlib.Path(__file__).parent
project_root = current_dir.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

ForcingType = Literal["P", "P_ET"]

# Nomes padrão das colunas brutas nos arquivos de entrada
DEFAULT_COLUMN_NAMES = {
    "flow":   "streamflow_m3s",
    "precip": "precipitation_chirps",
    "et":     "potential_evapotransp_gleam",
}


def _parse_column_names(
    column_names: Optional[Union[List[str], Dict[str, str]]]
) -> Dict[str, Optional[str]]:
    """
    Converte column_names (lista ou dict) para dict interno padronizado.

    Lista: [flow_col] ou [flow_col, precip_col] ou [flow_col, precip_col, et_col]
    Dict:  {'flow': ..., 'precip': ..., 'et': ...}  (chaves opcionais exceto 'flow')
    None:  usa DEFAULT_COLUMN_NAMES
    """
    if column_names is None:
        return dict(DEFAULT_COLUMN_NAMES)

    if isinstance(column_names, list):
        keys = ["flow", "precip", "et"]
        result = {k: None for k in keys}
        for i, val in enumerate(column_names):
            if i < len(keys):
                result[keys[i]] = val
        return result

    if isinstance(column_names, dict):
        # Suporta chaves antigas (precip_obs, precip_forecast) por retrocompatibilidade
        result = dict(DEFAULT_COLUMN_NAMES)
        if "flow" in column_names:
            result["flow"] = column_names["flow"]
        if "precip" in column_names:
            result["precip"] = column_names["precip"]
        # Retrocompatibilidade: precip_obs → precip (ignora precip_forecast, era igual)
        if "precip_obs" in column_names and "precip" not in column_names:
            result["precip"] = column_names["precip_obs"]
        if "et" in column_names:
            result["et"] = column_names["et"]
        if "et_obs" in column_names and "et" not in column_names:
            result["et"] = column_names["et_obs"]
        return result

    raise TypeError(f"column_names deve ser lista, dict ou None. Recebeu: {type(column_names)}")


class HydroFeatureEngineer:
    """
    Feature engineering para dados hidrológicos.

    Forcings:
      "P"    — apenas precipitação (padrão)
      "P_ET" — precipitação + evapotranspiração
    """

    def __init__(
        self,
        api_k_list: Optional[List[float]] = None,
        ma_windows: Optional[List[int]] = None,
        cumulative_windows: Optional[List[int]] = None,
        et_ma_windows: Optional[List[int]] = None,
        anomaly_ma_windows: Optional[List[int]] = None,
        forcings: ForcingType = "P",
        # Parâmetros legados — aceitos para não quebrar chamadas antigas, ignorados
        precipitation_ma_windows: Optional[List[int]] = None,
        precipitation_cumulative_windows: Optional[List[int]] = None,
        forecast_ma_windows: Optional[List[int]] = None,
        forecast_cumulative_windows: Optional[List[int]] = None,
        evapotranspiration_ma_windows: Optional[List[int]] = None,
    ):
        """
        Args:
            api_k_list: Valores de k para o Antecedent Precipitation Index.
            ma_windows: Janelas para médias móveis de precipitação (unificado).
            cumulative_windows: Janelas para acumulados de precipitação (unificado).
            et_ma_windows: Janelas para médias móveis de ET.
            anomaly_ma_windows: Janelas para médias móveis de anomalias.
            forcings: "P" ou "P_ET".
            precipitation_ma_windows: [LEGADO] alias de ma_windows.
            precipitation_cumulative_windows: [LEGADO] alias de cumulative_windows.
            forecast_ma_windows: [LEGADO] ignorado (série unificada).
            forecast_cumulative_windows: [LEGADO] ignorado (série unificada).
            evapotranspiration_ma_windows: [LEGADO] alias de et_ma_windows.
        """
        self.api_k_list = api_k_list or [0.70, 0.80, 0.85, 0.90, 0.92, 0.95]

        # Parâmetros legados como alias quando o novo não foi passado
        self.ma_windows = (
            ma_windows
            or precipitation_ma_windows
            or [3, 7, 15]
        )
        self.cumulative_windows = (
            cumulative_windows
            or precipitation_cumulative_windows
            or [3, 5, 7, 10]
        )
        self.et_ma_windows = (
            et_ma_windows
            or evapotranspiration_ma_windows
            or [7, 14, 30]
        )
        self.anomaly_ma_windows = anomaly_ma_windows or [3, 7]
        self.forcings = forcings or "P"

        if forecast_ma_windows or forecast_cumulative_windows:
            print(
                "⚠️  forecast_ma_windows / forecast_cumulative_windows não são mais usados "
                "(série unificada). Parâmetros ignorados."
            )

    # ------------------------------------------------------------------
    # Métodos de features
    # ------------------------------------------------------------------

    @staticmethod
    def compute_api(series: pd.Series, k: float) -> pd.Series:
        """Calcula o Antecedent Precipitation Index (API) causal."""
        vals = series.to_numpy(dtype=np.float64)
        api = np.zeros_like(vals)
        for i in range(len(vals)):
            p = vals[i] if not np.isnan(vals[i]) else 0.0
            api[i] = p if i == 0 else (p + k * api[i - 1])
        return pd.Series(api, index=series.index)

    def add_precipitation_features(
        self,
        df: pd.DataFrame,
        station: int,
        precip_col: str,
    ) -> pd.DataFrame:
        """
        Adiciona médias móveis e acumulados de precipitação (série unificada).
        Prefixo de saída: `precipitation_ma{w}_{station}` e `precipitation_cum{w}_{station}`.
        """
        for w in self.ma_windows:
            df[f"precipitation_ma{w}_{station}"] = (
                df[precip_col].rolling(window=w, min_periods=max(1, w // 2)).mean()
            )
        for w in self.cumulative_windows:
            df[f"precipitation_cum{w}_{station}"] = (
                df[precip_col].rolling(window=w, min_periods=max(1, w // 2)).sum()
            )
        return df

    def add_api_features(
        self,
        df: pd.DataFrame,
        station: int,
        precip_col: str,
    ) -> pd.DataFrame:
        """
        Adiciona API calculado sobre a série unificada de precipitação.
        Prefixo de saída: `api_k{kk}_{station}`.
        """
        for k in self.api_k_list:
            tag = f"k{int(round(k * 100)):02d}"
            df[f"api_{tag}_{station}"] = self.compute_api(df[precip_col], k)
        return df

    def add_et_features(
        self,
        df: pd.DataFrame,
        station: int,
        et_col: str,
    ) -> pd.DataFrame:
        """
        Adiciona médias móveis de ET (série unificada).
        Prefixo de saída: `et_ma{w}_{station}`.
        """
        for w in self.et_ma_windows:
            df[f"et_ma{w}_{station}"] = (
                df[et_col].rolling(window=w, min_periods=max(1, w // 2)).mean()
            )
        return df

    def add_anomaly_features(
        self,
        df: pd.DataFrame,
        station: int,
        anomaly_col: str,
    ) -> pd.DataFrame:
        """Adiciona médias móveis de anomalia logarítmica."""
        for w in self.anomaly_ma_windows:
            df[f"log_anomaly_ma{w}_{station}"] = (
                df[anomaly_col]
                .rolling(window=w, min_periods=max(1, w // 2))
                .mean()
                .shift(1)
                .fillna(0.0)
            )
        return df

    def add_advanced_features(
        self,
        df: pd.DataFrame,
        station: int,
        flow_col: str,
        precip_col: Optional[str],
        train_date_cutoff: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Adiciona dQ_dt, dP_dt (unificado), regime_state e log_anomaly.
        """
        # Período de referência para mediana sazonal
        ref_df = (
            df.loc[df.index <= pd.to_datetime(train_date_cutoff)]
            if train_date_cutoff is not None
            else df
        )
        seasonal_median = ref_df.groupby(ref_df.index.dayofyear)[flow_col].median()

        # Derivada do fluxo
        df[f"dQ_dt_{station}"] = df[flow_col].diff().fillna(0.0)

        # Derivada de precipitação (série unificada — um único dP_dt)
        if precip_col and precip_col in df.columns:
            df[f"dP_dt_{station}"] = df[precip_col].diff().fillna(0.0)

        # Estado do regime (baseado em dQ_dt suavizado)
        dQ_smooth = df[f"dQ_dt_{station}"].rolling(5, center=True, min_periods=1).mean()
        dQ_std_norm = (dQ_smooth - dQ_smooth.mean()) / (dQ_smooth.std() + 1e-6)
        regime_state = np.zeros(len(dQ_std_norm), dtype=np.int8)
        state, counter = 0, 0
        for i, val in enumerate(dQ_std_norm.to_numpy()):
            counter += 1
            if state <= 0 and val >= 0.2:
                state, counter = 1, 0
            elif state >= 0 and val <= -0.25:
                state, counter = -1, 0
            elif abs(val) < 0.1 and counter >= 3:
                state, counter = 0, 0
            regime_state[i] = state
        df[f"regime_state_{station}"] = regime_state

        # Anomalia logarítmica
        median_vals = (
            df.index.dayofyear.map(seasonal_median).fillna(0.0).to_numpy()
        )
        log_q = np.log1p(df[flow_col].to_numpy())
        log_median = np.log1p(median_vals)
        anomaly_col = f"log_anomaly_{station}"
        df[anomaly_col] = log_q - log_median
        df = self.add_anomaly_features(df, station, anomaly_col)

        return df

    def add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona codificações cíclicas de dia e mês."""
        df["day_sin"]   = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df["day_cos"]   = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12.0)
        return df

    # ------------------------------------------------------------------
    # Processamento por estação
    # ------------------------------------------------------------------

    def process_station(
        self,
        df: pd.DataFrame,
        station_id: int,
        train_date_cutoff: Optional[str] = None,
        column_names: Optional[Union[List[str], Dict[str, str]]] = None,
        forcings: Optional[ForcingType] = None,
    ) -> pd.DataFrame:
        """
        Processa uma estação e retorna DataFrame com todas as features.

        Args:
            df: DataFrame bruto com coluna 'date' (ou índice DatetimeIndex).
            station_id: ID numérico da estação.
            train_date_cutoff: Data de corte para calcular estatísticas de treino.
            column_names: Mapeamento das colunas brutas.
                - Lista: [flow_col] ou [flow_col, precip_col] ou
                         [flow_col, precip_col, et_col]
                - Dict:  {'flow': ..., 'precip': ..., 'et': ...}
                         (chaves 'precip_obs'/'et_obs' aceitas por retrocompat.)
                - None:  usa DEFAULT_COLUMN_NAMES
            forcings: "P" ou "P_ET". Se None, usa o da classe.
        """
        use_forcings = forcings if forcings is not None else self.forcings
        col_map = _parse_column_names(column_names)

        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df deve ter coluna 'date' ou índice DatetimeIndex.")

        # Verificar coluna obrigatória
        flow_raw = col_map.get("flow")
        if not flow_raw or flow_raw not in df.columns:
            raise ValueError(
                f"Estação {station_id}: coluna de vazão '{flow_raw}' não encontrada. "
                f"Disponíveis: {list(df.columns)}"
            )

        # ---- Renomear colunas para nomes internos ----
        rename_map: Dict[str, str] = {}
        rename_map[flow_raw] = f"Q_{station_id}"

        precip_raw = col_map.get("precip")
        if precip_raw and precip_raw in df.columns:
            rename_map[precip_raw] = f"precipitation_{station_id}"

        et_raw = col_map.get("et")
        if use_forcings == "P_ET" and et_raw and et_raw in df.columns:
            rename_map[et_raw] = f"et_{station_id}"

        df = df.rename(columns=rename_map)

        # Nomes internos
        flow_col   = f"Q_{station_id}"
        precip_col = f"precipitation_{station_id}" if f"precipitation_{station_id}" in df.columns else None
        et_col     = f"et_{station_id}" if (use_forcings == "P_ET" and f"et_{station_id}" in df.columns) else None

        # ---- Features de precipitação ----
        if precip_col:
            df = self.add_precipitation_features(df, station_id, precip_col)
            df = self.add_api_features(df, station_id, precip_col)

        # ---- Features de ET ----
        if et_col:
            df = self.add_et_features(df, station_id, et_col)

        # ---- Features avançadas (dQ_dt, dP_dt, regime, log_anomaly) ----
        df = self.add_advanced_features(
            df, station_id, flow_col, precip_col, train_date_cutoff
        )

        # ---- Manter apenas colunas desta estação + Q ----
        keep_cols = [c for c in df.columns if str(station_id) in c or c == flow_col]
        return df[keep_cols]

    def process_multiple_stations(
        self,
        data_dict: Dict[int, pd.DataFrame],
        train_date_cutoff: Optional[str] = None,
        column_names: Optional[Union[List[str], Dict[str, str]]] = None,
        forcings: Optional[ForcingType] = None,
    ) -> pd.DataFrame:
        """
        Processa múltiplas estações e retorna DataFrame combinado.

        Args:
            data_dict: {station_id: DataFrame bruto}.
            train_date_cutoff: Data de corte para estatísticas de treino.
            column_names: Ver process_station.
            forcings: "P" ou "P_ET". Se None, usa o da classe.
        """
        use_forcings = forcings if forcings is not None else self.forcings

        if not data_dict:
            raise ValueError("data_dict vazio.")

        print(f"{'='*60}")
        print(f"PROCESSANDO {len(data_dict)} ESTAÇÃO(ÕES)")
        print(f"   Forçantes: {use_forcings}")
        print(f"{'='*60}")

        processed_dfs = []
        for station_id, df in data_dict.items():
            try:
                print(f"📊 Estação {station_id}...")
                df_proc = self.process_station(
                    df=df,
                    station_id=station_id,
                    train_date_cutoff=train_date_cutoff,
                    column_names=column_names,
                    forcings=use_forcings,
                )
                processed_dfs.append(df_proc)
                print(f"✅ OK — {len(df_proc.columns)} features")
            except Exception as e:
                print(f"❌ Erro estação {station_id}: {e}")
                continue

        if not processed_dfs:
            raise ValueError("Nenhuma estação processada com sucesso.")

        # Combinar todas as estações
        combined_df = processed_dfs[0]
        for df in processed_dfs[1:]:
            combined_df = combined_df.merge(
                df, left_index=True, right_index=True, how="outer"
            )

        # Preenche gaps dentro do intervalo válido de CADA ESTAÇÃO (usando
        # Q_{station} como referência) e deixa NaN fora — as janelas fora
        # desses intervalos são descartadas em HydroDataset._build_valid_indices.
        #
        # Agrupar por sufixo "_{station_id}" garante que features com edges de
        # rolling-window (precipitation_ma_30, api_k, etc.) sejam preenchidas
        # dentro da janela válida da estação, evitando NaN na entrada do modelo.
        suffix_re = re.compile(r"_(\d+)$")
        station_cols: Dict[int, List[str]] = {}
        global_cols: List[str] = []
        for col in combined_df.columns:
            m = suffix_re.search(col)
            if m:
                station_cols.setdefault(int(m.group(1)), []).append(col)
            else:
                global_cols.append(col)

        for st, cols in station_cols.items():
            q_col = f"Q_{st}"
            if q_col not in combined_df.columns:
                continue
            q = combined_df[q_col]
            first, last = q.first_valid_index(), q.last_valid_index()
            if first is None or last is None:
                continue
            combined_df.loc[first:last, cols] = (
                combined_df.loc[first:last, cols].ffill().bfill()
            )

        for col in global_cols:
            s = combined_df[col]
            first, last = s.first_valid_index(), s.last_valid_index()
            if first is None or last is None:
                continue
            combined_df.loc[first:last, col] = s.loc[first:last].ffill().bfill()

        combined_df = self.add_seasonal_features(combined_df)

        print(f"✅ CONCLUÍDO — {len(combined_df.columns)} colunas")
        print(f"{'='*60}")

        return combined_df


# ------------------------------------------------------------------
# Funções auxiliares de I/O
# ------------------------------------------------------------------

def load_station_data(
    complete_series_dir: pathlib.Path,
    station_ids: List[int],
) -> Dict[int, pd.DataFrame]:
    """Carrega CSVs de séries completas por estação."""
    data_dict: Dict[int, pd.DataFrame] = {}
    for station_id in station_ids:
        file_path = complete_series_dir / f"{station_id}_complete_date.csv"
        if file_path.exists():
            try:
                data_dict[station_id] = pd.read_csv(file_path)
            except Exception as e:
                print(f"✗ Erro ao carregar estação {station_id}: {e}")
        else:
            print(f"✗ Arquivo não encontrado: {file_path}")
    return data_dict


# Alias para retrocompatibilidade
load_observed_data = load_station_data


def load_forecast_data(
    forecast_dir: pathlib.Path,
    station_ids: List[int],
    forcings: ForcingType = "P",
) -> Dict[int, pd.DataFrame]:
    """
    Carrega CSVs de forecast por estação.
    Retorna dict {station_id: DataFrame com 'date' e 'precipitation_forecast' [e 'et_forecast']}.
    """
    forecast_dict: Dict[int, pd.DataFrame] = {}
    need_et = forcings == "P_ET"

    for station_id in station_ids:
        precip_file = forecast_dir / f"{station_id}_precipitation_forecast.csv"
        et_file     = forecast_dir / f"{station_id}_evapotranspiration_forecast.csv"

        if not precip_file.exists():
            print(f"✗ Forecast de precipitação não encontrado: {station_id}")
            continue
        if need_et and not et_file.exists():
            print(f"⚠️  Forecast de ET não encontrado para estação {station_id}")
            continue

        try:
            df_precip = pd.read_csv(precip_file, parse_dates=["date"])
            df_precip["date"] = pd.to_datetime(df_precip["date"])
            df_precip = df_precip[["date", "precipitation_forecast"]]

            if need_et:
                df_et = pd.read_csv(et_file, parse_dates=["date"])
                df_et["date"] = pd.to_datetime(df_et["date"])
                df_et = df_et[["date", "et_forecast"]]
                df_precip = df_precip.merge(df_et, on="date", how="outer")

            forecast_dict[station_id] = df_precip
        except Exception as e:
            print(f"✗ Erro ao carregar forecast da estação {station_id}: {e}")

    return forecast_dict


def merge_observed_and_forecast(
    observed_dict: Dict[int, pd.DataFrame],
    forecast_dict: Dict[int, pd.DataFrame],
    forcings: ForcingType = "P",
    obs_precip_col: str = "precipitation_chirps",
    obs_et_col: str = "potential_evapotransp_gleam",
) -> Dict[int, pd.DataFrame]:
    """
    Combina séries observadas e de forecast em série contínua unificada.

    A coluna de precipitação de saída é `precipitation` (sem sufixo _obs/_forecast):
      - Valores do período observado vêm de `obs_precip_col`.
      - Gaps são preenchidos com `precipitation_forecast` do forecast_dict.

    Para ET (forcings="P_ET"):
      - Coluna de saída é `et` (unificada).

    Args:
        observed_dict: {station_id: DataFrame observado}.
        forecast_dict: {station_id: DataFrame forecast}.
        forcings: "P" ou "P_ET".
        obs_precip_col: Nome da coluna de precipitação observada no CSV bruto.
        obs_et_col: Nome da coluna de ET observada no CSV bruto.
    """
    merged_dict: Dict[int, pd.DataFrame] = {}
    need_et = forcings == "P_ET"

    for station_id, df_obs in observed_dict.items():
        df_obs = df_obs.copy()
        df_obs["date"] = pd.to_datetime(df_obs["date"])

        # ---- Precipitação unificada ----
        if obs_precip_col in df_obs.columns:
            df_obs["precipitation"] = df_obs[obs_precip_col]
        else:
            available = [c for c in df_obs.columns if "precip" in c.lower() or "chirps" in c.lower()]
            if available:
                print(
                    f"⚠️  Estação {station_id}: coluna '{obs_precip_col}' não encontrada. "
                    f"Usando '{available[0]}' como precipitação."
                )
                df_obs["precipitation"] = df_obs[available[0]]
            else:
                print(f"⚠️  Estação {station_id}: sem coluna de precipitação. Preenchendo com 0.")
                df_obs["precipitation"] = 0.0

        # Preencher gaps de precipitação com forecast (se disponível)
        if station_id in forecast_dict:
            df_fc = forecast_dict[station_id].copy()
            df_fc["date"] = pd.to_datetime(df_fc["date"])
            df_merged = df_obs.merge(df_fc, on="date", how="left")

            if "precipitation_forecast" in df_merged.columns:
                df_merged["precipitation"] = df_merged["precipitation"].fillna(
                    df_merged["precipitation_forecast"]
                )
                df_merged = df_merged.drop(columns=["precipitation_forecast"])
        else:
            df_merged = df_obs.copy()

        # ---- ET unificada (apenas se forcings="P_ET") ----
        if need_et:
            if obs_et_col in df_merged.columns:
                df_merged["et"] = df_merged[obs_et_col]
                # Preencher com et_forecast se disponível
                if "et_forecast" in df_merged.columns:
                    df_merged["et"] = df_merged["et"].fillna(df_merged["et_forecast"])
                    df_merged = df_merged.drop(columns=["et_forecast"])
            else:
                print(f"⚠️  Estação {station_id}: coluna ET '{obs_et_col}' não encontrada.")

        # Remover colunas brutas de obs que não serão mais usadas
        cols_to_drop = [
            c for c in df_merged.columns
            if c in [obs_precip_col, obs_et_col, "precipitation_forecast", "et_forecast"]
            or c.endswith("_x") or c.endswith("_y")
            or c in ["station_id", "missing_data", "Unnamed: 0"]
        ]
        df_merged = df_merged.drop(columns=[c for c in cols_to_drop if c in df_merged.columns])

        # Remover colunas de ET quando forcings="P" para não poluir o df
        if not need_et:
            et_cols = [
                c for c in df_merged.columns
                if any(kw in c.lower() for kw in ["evapotransp", "gleam", "et_obs", "et_fc", "_et_"])
            ]
            if et_cols:
                df_merged = df_merged.drop(columns=et_cols)

        merged_dict[station_id] = df_merged

    return merged_dict


# ------------------------------------------------------------------
# Função principal de processamento
# ------------------------------------------------------------------

def process_features(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    station_ids: List[int],
    forecast_dir: Optional[pathlib.Path] = None,
    api_k_list: Optional[List[float]] = None,
    ma_windows: Optional[List[int]] = None,
    cumulative_windows: Optional[List[int]] = None,
    et_ma_windows: Optional[List[int]] = None,
    anomaly_ma_windows: Optional[List[int]] = None,
    train_date_cutoff: Optional[str] = None,
    output_filename: str = "features_combined.csv",
    forcings: ForcingType = "P",
    column_names: Optional[Union[List[str], Dict[str, str]]] = None,
    # Parâmetros legados — aceitos para retrocompatibilidade, mapeados internamente
    precipitation_ma_windows: Optional[List[int]] = None,
    precipitation_cumulative_windows: Optional[List[int]] = None,
    forecast_ma_windows: Optional[List[int]] = None,
    forecast_cumulative_windows: Optional[List[int]] = None,
    evapotranspiration_ma_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Função principal de processamento de features para treino/validação/teste.

    Args:
        input_dir: Diretório com CSVs de séries completas observadas.
        output_dir: Diretório de saída.
        forecast_dir: Diretório com CSVs de forecast. Opcional. Se None (ou
                      o diretório não existir), o pipeline roda apenas com
                      a precipitação observada — o próprio valor observado
                      em D+0…D+N é tratado como "forecast" pelo modelo.
        station_ids: IDs das estações.
        api_k_list: Valores de k para API.
        ma_windows: Janelas de médias móveis de precipitação.
        cumulative_windows: Janelas de acumulados de precipitação.
        et_ma_windows: Janelas de médias móveis de ET.
        anomaly_ma_windows: Janelas de médias móveis de anomalia.
        train_date_cutoff: Data de corte para estatísticas de treino.
        output_filename: Nome do arquivo CSV de saída.
        forcings: "P" ou "P_ET".
        column_names: Mapeamento de colunas (lista ou dict). Se None, usa padrões.
        precipitation_ma_windows: [LEGADO] alias de ma_windows.
        precipitation_cumulative_windows: [LEGADO] alias de cumulative_windows.
        forecast_ma_windows: [LEGADO] ignorado.
        forecast_cumulative_windows: [LEGADO] ignorado.
        evapotranspiration_ma_windows: [LEGADO] alias de et_ma_windows.
    """
    print("=" * 60)
    print("PROCESSAMENTO DE FEATURES")
    print(f"   Forçantes: {forcings}")
    print("=" * 60)

    if not input_dir.exists():
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 Carregando dados observados de {len(station_ids)} estações...")
    observed_dict = load_station_data(input_dir, station_ids)
    print(f"✓ {len(observed_dict)} estações carregadas")

    if forecast_dir is not None and forecast_dir.exists():
        print("🔮 Carregando dados de forecast...")
        forecast_dict = load_forecast_data(forecast_dir, station_ids, forcings=forcings)
        print(f"✓ {len(forecast_dict)} estações com forecast carregadas")
    else:
        forecast_dict = {}
        print("ℹ️  Sem diretório de forecast — usando precipitação observada como "
              "fonte única (D+0…D+N tratado como 'forecast' pelo modelo).")

    # Extrair nomes brutos das colunas obs para o merge
    # (antes do merge, os nomes ainda são os originais do CSV)
    col_map_raw = _parse_column_names(column_names)
    obs_precip_col = col_map_raw.get("precip") or "precipitation_chirps"
    obs_et_col     = col_map_raw.get("et")     or "potential_evapotransp_gleam"

    print("🔗 Criando série unificada obs + forecast...")
    merged_dict = merge_observed_and_forecast(
        observed_dict, forecast_dict, forcings=forcings,
        obs_precip_col=obs_precip_col,
        obs_et_col=obs_et_col,
    )
    print(f"✓ {len(merged_dict)} estações combinadas")

    if not merged_dict:
        raise ValueError("Nenhum dado foi combinado. Verifique os arquivos.")

    # Após o merge, as colunas unificadas têm sempre os mesmos nomes:
    # 'precipitation' (e 'et' se P_ET) — independente dos nomes brutos originais.
    internal_col_names = {"flow": "streamflow_m3s", "precip": "precipitation"}
    if forcings == "P_ET":
        internal_col_names["et"] = "et"

    print("⚙️  Criando features...")
    engineer = HydroFeatureEngineer(
        api_k_list=api_k_list,
        ma_windows=ma_windows or precipitation_ma_windows,
        cumulative_windows=cumulative_windows or precipitation_cumulative_windows,
        et_ma_windows=et_ma_windows or evapotranspiration_ma_windows,
        anomaly_ma_windows=anomaly_ma_windows,
        forcings=forcings,
        forecast_ma_windows=forecast_ma_windows,
        forecast_cumulative_windows=forecast_cumulative_windows,
    )

    combined_df = engineer.process_multiple_stations(
        merged_dict,
        train_date_cutoff=train_date_cutoff,
        column_names=internal_col_names,
        forcings=forcings,
    )

    output_path = output_dir / output_filename
    combined_df.to_csv(output_path)

    print("\n" + "=" * 60)
    print("✅ FEATURES CRIADAS COM SUCESSO")
    print("=" * 60)
    print(f"  Forçantes:            {forcings}")
    print(f"  Estações processadas: {len(merged_dict)}")
    print(f"  Período:              {combined_df.index.min().date()} a {combined_df.index.max().date()}")
    print(f"  Total de dias:        {len(combined_df)}")
    print(f"  Total de features:    {len(combined_df.columns)}")
    print(f"  Arquivo salvo:        {output_path}")
    print("=" * 60)

    return combined_df
