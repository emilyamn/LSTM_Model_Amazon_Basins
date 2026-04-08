"""
Módulo de Processamento de Inferência Hidrológica

Mudanças principais (v2 — alinhado com features_processing.py v2):
- Série de precipitação UNIFICADA: coluna `precipitation` contínua
  (obs até reference_date, forecast real após reference_date).
- ET também unificada em `et` quando forcings="P_ET".
- Todas as features derivadas (ma, cum, api, dP_dt) calculadas sobre a
  série unificada — sem mais colunas _obs / _forecast duplicadas.
- _apply_forecast_mask simplificado: com série unificada, não há colunas
  _forecast para mascarar. Apenas Q_{station} fica NaN após d0 por design.
"""

from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field
import pathlib
import sys
import pandas as pd
import numpy as np

try:
    from .features_processing import HydroFeatureEngineer
except ImportError:
    from features_processing import HydroFeatureEngineer

ForcingType = Literal["P", "P_ET"]

current_dir = pathlib.Path(__file__).parent
project_root = current_dir.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# ==============================================================================
# DATACLASSES DE CONFIGURAÇÃO
# ==============================================================================

@dataclass
class InferenceConfig:
    """Configuração para processamento de inferência."""
    observed_inference_dir: pathlib.Path
    forecast_inference_dir: pathlib.Path
    output_dir: pathlib.Path
    station_ids: List[int]
    reference_dates: List[str]
    column_mapping: Optional[Dict[str, Optional[str]]] = None
    forcings: Literal["P", "P_ET"] = "P"
    api_k_list: Optional[List[float]] = None
    # Parâmetros de janelas — nomes canônicos
    ma_windows: Optional[List[int]] = None
    cumulative_windows: Optional[List[int]] = None
    et_ma_windows: Optional[List[int]] = None
    anomaly_ma_windows: Optional[List[int]] = None
    output_filename: str = "features_combined_inference.csv"
    # Parâmetros legados — aceitos para retrocompatibilidade, mapeados internamente
    precipitation_ma_windows: Optional[List[int]] = None
    precipitation_cumulative_windows: Optional[List[int]] = None
    forecast_ma_windows: Optional[List[int]] = None         # ignorado (série unificada)
    forecast_cumulative_windows: Optional[List[int]] = None  # ignorado (série unificada)
    evapotranspiration_ma_windows: Optional[List[int]] = None


# Nomes padrão das colunas brutas nos arquivos de entrada de inferência
INTERNAL_COLUMN_NAMES = {
    "date":   "date",
    "flow":   "streamflow_m3s",
    "precip": "precipitation_chirps",
    "et":     "potential_evapotransp_gleam",
}


# ==============================================================================
# FUNÇÃO PRINCIPAL
# ==============================================================================

def process_inference(
    config: InferenceConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Processa dados de inferência gerando DataFrame de features unificado.

    Fluxo:
      1. Carrega séries observadas (até reference_date)
      2. Carrega forecast de precipitação (após reference_date)
      3. Cria série UNIFICADA contínua por estação
      4. Gera features hidrológicas sobre a série unificada
      5. Salva e retorna DataFrame de features
    """
    if verbose:
        print("=" * 60)
        print("PROCESSAMENTO DE INFERÊNCIA HIDROLÓGICA")
        print("=" * 60)
        print(f"📅 Reference dates: {config.reference_dates}")
        print(f"🏭 Estações: {config.station_ids}")
        print(f"🔧 Forçantes: {config.forcings}")

    column_mapping = _prepare_column_mapping(config.column_mapping)
    if config.forcings == "P":
        column_mapping["et"] = None

    if verbose:
        print("📋 Mapeamento de colunas:")
        for key, value in column_mapping.items():
            label = value if value else "NÃO INCLUÍDO"
            print(f"   {key} -> {label}")

    if not config.observed_inference_dir.exists():
        raise FileNotFoundError(
            f"Diretório de observados não encontrado: {config.observed_inference_dir}"
        )
    if not config.forecast_inference_dir.exists():
        if verbose:
            print("⚠️  Diretório de forecast não encontrado — usando apenas observados")
        config.forecast_inference_dir = config.observed_inference_dir

    config.output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("📂 Carregando dados observados...")
    observed_dict = _load_observed_data(
        config.observed_inference_dir, config.station_ids, column_mapping, verbose
    )
    if not observed_dict:
        raise ValueError("Nenhum dado observado foi carregado.")

    if verbose:
        print("📂 Carregando dados de forecast...")
    forecast_dict = _load_forecast_data(
        config.forecast_inference_dir, config.station_ids, column_mapping,
        config.forcings, verbose
    )

    if verbose:
        print("🔗 Criando série contínua unificada...")
    merged_dict = _create_unified_series(
        observed_dict=observed_dict,
        forecast_dict=forecast_dict,
        station_ids=config.station_ids,
        reference_dates=config.reference_dates,
        forcings=config.forcings,
        verbose=verbose,
    )

    if verbose:
        print("⚙️  Gerando features hidrológicas...")
    df_features = _generate_features(merged_dict, config, verbose)

    output_path = config.output_dir / config.output_filename
    df_features.to_csv(output_path)

    if verbose:
        print("=" * 60)
        print("✅ INFERÊNCIA PROCESSADA COM SUCESSO")
        print("=" * 60)
        print(f"   Forçantes:         {config.forcings}")
        print(f"   Período:           {df_features.index.min().date()} a {df_features.index.max().date()}")
        print(f"   Total de registros:{len(df_features)}")
        print(f"   Total de features: {len(df_features.columns)}")
        print(f"   Arquivo salvo:     {output_path}")
        print("=" * 60)

    return df_features


# ==============================================================================
# CRIAÇÃO DA SÉRIE UNIFICADA
# ==============================================================================

def _create_unified_series(
    observed_dict: Dict[int, pd.DataFrame],
    forecast_dict: Dict[int, pd.DataFrame],
    station_ids: List[int],
    reference_dates: List[str],
    forcings: str = "P",
    verbose: bool = True,
) -> Dict[int, pd.DataFrame]:
    """
    Para cada estação, cria um DataFrame com:
      - 'date', 'streamflow_m3s': obs até reference_date, NaN depois
      - 'precipitation': obs até reference_date, forecast depois  ← UNIFICADA
      - 'et' (se P_ET): obs até reference_date, forecast depois   ← UNIFICADA
    """
    merged_dict: Dict[int, pd.DataFrame] = {}
    max_ref_date = pd.to_datetime(reference_dates).max()
    need_et = forcings == "P_ET"

    for station_id in station_ids:
        if station_id not in observed_dict:
            continue

        df_obs = observed_dict[station_id].copy()
        df_obs["date"] = pd.to_datetime(df_obs["date"])

        if verbose:
            print(f"   📊 Estação {station_id}:")
            print(f"      Observados:     {df_obs['date'].min().date()} até {df_obs['date'].max().date()}")
            print(f"      Reference date: {max_ref_date.date()}")

        # Iniciar com série obs
        df_unified = df_obs.copy()

        # ── Precipitação unificada ────────────────────────────────────────────
        # Renomear coluna obs para 'precipitation' (coluna unificada)
        if "precipitation_chirps" in df_unified.columns:
            df_unified = df_unified.rename(columns={"precipitation_chirps": "precipitation"})
        elif "precipitation" not in df_unified.columns:
            df_unified["precipitation"] = np.nan

        if station_id in forecast_dict:
            df_fc = forecast_dict[station_id].copy()
            df_fc["date"] = pd.to_datetime(df_fc["date"])

            if verbose:
                print(f"      Forecast:       {df_fc['date'].min().date()} até {df_fc['date'].max().date()}")

            # Merge com forecast
            df_unified = df_unified.merge(df_fc, on="date", how="outer")
            df_unified = df_unified.sort_values("date").reset_index(drop=True)

            # Preencher precipitação após reference_date com forecast
            if "precipitation_forecast" in df_unified.columns:
                mask_future = df_unified["date"] > max_ref_date
                df_unified.loc[mask_future, "precipitation"] = (
                    df_unified.loc[mask_future, "precipitation_forecast"]
                )
                df_unified = df_unified.drop(columns=["precipitation_forecast"])

            # ET unificada (se P_ET)
            if need_et and "et_forecast" in df_unified.columns:
                if "et" not in df_unified.columns:
                    df_unified["et"] = np.nan
                mask_future = df_unified["date"] > max_ref_date
                df_unified.loc[mask_future, "et"] = (
                    df_unified.loc[mask_future, "et_forecast"]
                )
                df_unified = df_unified.drop(columns=["et_forecast"])

        else:
            if verbose:
                print("      ⚠️  Sem forecast — precipitação NaN após reference_date")
            mask_future = df_unified["date"] > max_ref_date
            df_unified.loc[mask_future, "precipitation"] = np.nan

        # Vazão: NaN após reference_date (futuro desconhecido)
        mask_future = df_unified["date"] > max_ref_date
        df_unified.loc[mask_future, "streamflow_m3s"] = np.nan

        # Remover colunas brutas residuais
        cols_to_drop = [
            c for c in df_unified.columns
            if c in ["precipitation_chirps", "potential_evapotransp_gleam",
                     "precipitation_forecast", "et_forecast"]
            or c.endswith("_x") or c.endswith("_y")
            or c in ["station_id", "missing_data", "Unnamed: 0"]
        ]
        df_unified = df_unified.drop(columns=[c for c in cols_to_drop if c in df_unified.columns])

        # Colunas finais esperadas pelo HydroFeatureEngineer
        keep = ["date", "streamflow_m3s", "precipitation"]
        if need_et and "et" in df_unified.columns:
            keep.append("et")
        df_unified = df_unified[[c for c in keep if c in df_unified.columns]]

        obs_count  = (df_unified["date"] <= max_ref_date).sum()
        fc_count   = (df_unified["date"] >  max_ref_date).sum()
        if verbose:
            print(f"      ✅ Série unificada: {obs_count} dias obs + {fc_count} dias forecast")

        merged_dict[station_id] = df_unified

    return merged_dict


# ==============================================================================
# GERAÇÃO DE FEATURES
# ==============================================================================

def _generate_features(
    merged_dict: Dict[int, pd.DataFrame],
    config: InferenceConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """Instancia HydroFeatureEngineer e processa todas as estações."""

    if not merged_dict:
        raise ValueError("merged_dict vazio — sem dados para processar.")

    # Resolver parâmetros legados
    ma_windows = (
        config.ma_windows
        or config.precipitation_ma_windows
        or [3, 7, 15]
    )
    cumulative_windows = (
        config.cumulative_windows
        or config.precipitation_cumulative_windows
        or [3, 5, 7, 10]
    )
    et_ma_windows = (
        config.et_ma_windows
        or config.evapotranspiration_ma_windows
        or [7, 14, 30]
    )

    engineer = HydroFeatureEngineer(
        api_k_list=config.api_k_list,
        ma_windows=ma_windows,
        cumulative_windows=cumulative_windows,
        et_ma_windows=et_ma_windows,
        anomaly_ma_windows=config.anomaly_ma_windows,
        forcings=config.forcings,
        # Parâmetros legados explicitamente ignorados
        forecast_ma_windows=None,
        forecast_cumulative_windows=None,
    )

    # column_names para HydroFeatureEngineer:
    # após _create_unified_series as colunas já têm nome padrão
    column_names = {
        "flow":   "streamflow_m3s",
        "precip": "precipitation",
    }
    if config.forcings == "P_ET":
        column_names["et"] = "et"

    if verbose:
        sample = list(merged_dict.values())[0]
        print(f"   📋 Forçantes:   {config.forcings}")
        print(f"   📋 Colunas:     {list(sample.columns)}")
        print(f"   📋 column_names:{column_names}")

    df_features = engineer.process_multiple_stations(
        merged_dict,
        train_date_cutoff=None,
        column_names=column_names,
        forcings=config.forcings,
    )

    # Restaurar NaN em Q e features derivadas de fluxo após reference_date.
    # O ffill().bfill() de process_multiple_stations propaga o último valor
    # observado — precisamos desfazer isso para as colunas que dependem de Q.
    max_ref_date = pd.to_datetime(config.reference_dates).max()
    mask_future = df_features.index > max_ref_date

    if mask_future.any():
        q_derived_prefixes = ("Q_", "dQ_dt_", "regime_state_", "log_anomaly")
        cols_to_nan = [
            c for c in df_features.columns
            if any(c.startswith(p) for p in q_derived_prefixes)
        ]
        df_features.loc[mask_future, cols_to_nan] = np.nan

        if verbose:
            print(
                f"   🔒 {len(cols_to_nan)} colunas derivadas de Q restauradas "
                f"para NaN após {max_ref_date.date()}"
            )

    return df_features


# ==============================================================================
# FUNÇÕES DE CARREGAMENTO
# ==============================================================================

def _prepare_column_mapping(
    user_mapping: Optional[Dict[str, Optional[str]]]
) -> Dict[str, Optional[str]]:
    mapping = INTERNAL_COLUMN_NAMES.copy()
    if user_mapping:
        for key, value in user_mapping.items():
            mapping[key] = value
    return mapping


def _load_observed_data(
    data_dir: pathlib.Path,
    station_ids: List[int],
    column_mapping: Dict[str, Optional[str]],
    verbose: bool = True,
) -> Dict[int, pd.DataFrame]:
    """Carrega séries observadas por estação."""
    observed_dict: Dict[int, pd.DataFrame] = {}

    for station_id in station_ids:
        file_path = _find_file(
            data_dir,
            [
                f"{station_id}_complete_date_inference.csv",
                f"{station_id}.csv",
                f"station_{station_id}.csv",
            ],
        )
        if file_path is None:
            if verbose:
                print(f"   ⚠️  Arquivo não encontrado para estação {station_id}")
            continue

        try:
            df = pd.read_csv(file_path)
            df = _standardize_observed_columns(df, column_mapping, station_id)
            observed_dict[station_id] = df
            if verbose:
                print(
                    f"   ✅ Estação {station_id}: {len(df)} registros "
                    f"(até {df['date'].max().date()})"
                )
        except Exception as e:
            if verbose:
                print(f"   ❌ Erro ao carregar {station_id}: {e}")

    return observed_dict


def _standardize_observed_columns(
    df: pd.DataFrame,
    column_mapping: Dict[str, Optional[str]],
    station_id: int,
) -> pd.DataFrame:
    """Renomeia colunas brutas para nomes internos padronizados."""
    df = df.copy()

    # Date
    date_raw = _find_column(df, ["date", "data", "time", "datetime"], required=True)
    df = df.rename(columns={date_raw: "date"})
    df["date"] = pd.to_datetime(df["date"])

    # Flow
    flow_raw = column_mapping.get("flow") or _find_column(
        df, ["streamflow_m3s", "flow", "vazao", "q", "mean_Q", "discharge"], required=False
    )
    if not flow_raw or flow_raw not in df.columns:
        raise ValueError(f"Estação {station_id}: coluna de vazão não encontrada.")
    df = df.rename(columns={flow_raw: "streamflow_m3s"})

    # Precipitation obs
    precip_raw = column_mapping.get("precip") or _find_column(
        df,
        ["precipitation_chirps", "precipitation_obs", "precipitation", "precip", "mean_P", "P"],
        required=False,
    )
    if precip_raw and precip_raw in df.columns:
        df = df.rename(columns={precip_raw: "precipitation_chirps"})

    # ET obs
    et_raw = column_mapping.get("et")
    if et_raw and et_raw in df.columns:
        df = df.rename(columns={et_raw: "potential_evapotransp_gleam"})

    # Manter apenas colunas relevantes
    keep = ["date", "streamflow_m3s"]
    for extra in ["precipitation_chirps", "potential_evapotransp_gleam"]:
        if extra in df.columns:
            keep.append(extra)

    return df[keep].sort_values("date").reset_index(drop=True)


def _load_forecast_data(
    data_dir: pathlib.Path,
    station_ids: List[int],
    column_mapping: Dict[str, Optional[str]],
    forcings: str = "P",
    verbose: bool = True,
) -> Dict[int, pd.DataFrame]:
    """
    Carrega CSVs de forecast de precipitação (e ET se P_ET) por estação.
    Retorna dict {station_id: DataFrame com 'date', 'precipitation_forecast' [, 'et_forecast']}.
    """
    forecast_dict: Dict[int, pd.DataFrame] = {}
    need_et = forcings == "P_ET"

    for station_id in station_ids:
        file_path = _find_file(
            data_dir,
            [
                f"{station_id}_precipitation_forecast.csv",
                f"{station_id}_forecast.csv",
            ],
        )
        if file_path is None:
            if verbose:
                print(f"   ⚠️  Sem forecast para estação {station_id}")
            continue

        try:
            df = pd.read_csv(file_path)
            date_raw = _find_column(df, ["date", "data"], required=True)
            df["date"] = pd.to_datetime(df[date_raw])

            # Precipitação forecast
            precip_raw = (
                column_mapping.get("precip")
                or _find_column(
                    df,
                    ["precipitation_forecast", "precipitation", "precip", "mean_P", "P"],
                    required=False,
                )
            )
            if not precip_raw or precip_raw not in df.columns:
                if verbose:
                    print(f"   ⚠️  Coluna de precipitação não encontrada no forecast de {station_id}")
                continue

            result = df[["date", precip_raw]].rename(
                columns={precip_raw: "precipitation_forecast"}
            )

            # ET forecast (se P_ET)
            if need_et:
                et_raw = column_mapping.get("et") or _find_column(
                    df,
                    ["et_forecast", "evapotranspiration_forecast", "et", "potential_evapotransp_gleam"],
                    required=False,
                )
                if et_raw and et_raw in df.columns:
                    result["et_forecast"] = df[et_raw].values
                elif verbose:
                    print(f"   ⚠️  Coluna de ET não encontrada no forecast de {station_id}")

            forecast_dict[station_id] = result
            if verbose:
                print(
                    f"   ✅ Forecast {station_id}: {len(result)} registros "
                    f"({result['date'].min().date()} até {result['date'].max().date()})"
                )

        except Exception as e:
            if verbose:
                print(f"   ❌ Erro forecast {station_id}: {e}")

    return forecast_dict


# ==============================================================================
# UTILITÁRIOS
# ==============================================================================

def _find_file(
    directory: pathlib.Path,
    candidates: List[str],
) -> Optional[pathlib.Path]:
    """Retorna o primeiro arquivo da lista que existir no diretório."""
    for name in candidates:
        p = directory / name
        if p.exists():
            return p
    return None


def _find_column(
    df: pd.DataFrame,
    possible_names: List[str],
    required: bool = False,
) -> Optional[str]:
    """Busca insensível a maiúsculas entre os nomes candidatos."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in possible_names:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    if required:
        raise ValueError(
            f"Coluna não encontrada. Tentativas: {possible_names}. "
            f"Disponíveis: {list(df.columns)}"
        )
    return None