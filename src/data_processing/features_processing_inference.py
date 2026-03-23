"""
Módulo de Processamento de Inferência Hidrológica
"""

from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
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
    precipitation_ma_windows: Optional[List[int]] = None
    precipitation_cumulative_windows: Optional[List[int]] = None
    forecast_ma_windows: Optional[List[int]] = None
    forecast_cumulative_windows: Optional[List[int]] = None
    evapotranspiration_ma_windows: Optional[List[int]] = None
    anomaly_ma_windows: Optional[List[int]] = None
    output_filename: str = "features_combined_inference.csv"


INTERNAL_COLUMN_NAMES = {
    "date": "date",
    "flow": "streamflow_m3s",
    "precip": "precipitation_chirps",
    "et": "potential_evapotransp_gleam"
}


# ==============================================================================
# FUNÇÃO PRINCIPAL DE PROCESSAMENTO
# ==============================================================================

def process_inference(
    config: InferenceConfig,
    verbose: bool = True
) -> pd.DataFrame:
    if verbose:
        print("=" * 60)
        print("PROCESSAMENTO DE INFERÊNCIA HIDROLÓGICA")
        print("=" * 60)
        print(f"📅 Reference dates: {config.reference_dates}")
        print(f"🏭 Estações: {config.station_ids}")
        print(f"🔧 Forçantes: {config.forcings}")

    column_mapping = _prepare_column_mapping(config.column_mapping)

    # ET: apenas se forcings="P_ET"
    if config.forcings == "P":
        column_mapping['et'] = None

    if verbose:
        print("📋 Mapeamento de colunas:")
        for key, value in column_mapping.items():
            if value is None:
                print(f"   {key}: NÃO INCLUÍDO")
            else:
                print(f"   {key} -> {value}")

    if not config.observed_inference_dir.exists():
        raise FileNotFoundError(f"Diretório de observados não encontrado: {config.observed_inference_dir}")

    if not config.forecast_inference_dir.exists():
        if verbose:
            print("⚠️ Diretório de forecast não encontrado")
        config.forecast_inference_dir = config.observed_inference_dir

    config.output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("📂 Carregando dados observados...")

    observed_dict = _load_observed_data(
        data_dir=config.observed_inference_dir,
        station_ids=config.station_ids,
        column_mapping=column_mapping,
        verbose=verbose
    )

    if not observed_dict:
        raise ValueError("Nenhum dado observado foi carregado.")

    if verbose:
        print("📂 Carregando dados de forecast...")

    forecast_dict = _load_forecast_precipitation(
        data_dir=config.forecast_inference_dir,
        station_ids=config.station_ids,
        column_mapping=column_mapping,
        verbose=verbose
    )

    if verbose:
        print("🔗 Criando série contínua...")

    merged_dict = _create_continuous_series_with_reference(
        observed_dict=observed_dict,
        forecast_dict=forecast_dict,
        station_ids=config.station_ids,
        reference_dates=config.reference_dates,
        verbose=verbose
    )

    if verbose:
        print("⚙️ Gerando features hidrológicas...")

    df_features = _generate_features(
        merged_dict=merged_dict,
        config=config,
        verbose=verbose
    )

    output_path = config.output_dir / config.output_filename
    df_features.to_csv(output_path)

    if verbose:
        print("=" * 60)
        print("✅ INFERÊNCIA PROCESSADA COM SUCESSO")
        print("=" * 60)
        print(f"   Forçantes: {config.forcings}")
        print(f"   Período: {df_features.index.min().date()} a {df_features.index.max().date()}")
        print(f"   Total de registros: {len(df_features)}")
        print(f"   Total de features: {len(df_features.columns)}")
        print(f"   Arquivo salvo: {output_path}")
        print("=" * 60)

    return df_features


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def _prepare_column_mapping(user_mapping: Optional[Dict[str, Optional[str]]]) -> Dict[str, Optional[str]]:
    mapping = INTERNAL_COLUMN_NAMES.copy()
    if user_mapping is not None:
        for key, value in user_mapping.items():
            if value is not None:
                mapping[key] = value
            else:
                mapping[key] = None
    return mapping


def _load_observed_data(data_dir, station_ids, column_mapping, verbose=True):
    observed_dict = {}
    for station_id in station_ids:
        possible_files = [
            data_dir / f"{station_id}_complete_date_inference.csv",
            data_dir / f"{station_id}.csv",
            data_dir / f"station_{station_id}.csv",
        ]
        file_path = None
        for pf in possible_files:
            if pf.exists():
                file_path = pf
                break
        if file_path is None:
            if verbose:
                print(f"   ⚠️ Arquivo não encontrado para estação {station_id}")
            continue
        try:
            df = pd.read_csv(file_path)
            df = _process_observed_columns(df, column_mapping, station_id, verbose)
            observed_dict[station_id] = df
            if verbose:
                print(f"   ✅ Estação {station_id}: {len(df)} registros (até {df['date'].max().date()})")
        except Exception as e:
            if verbose:
                print(f"   ❌ Erro ao carregar {station_id}: {e}")
            continue
    return observed_dict


def _process_observed_columns(df, column_mapping, station_id, verbose=True):
    df = df.copy()
    date_col = _find_column(df, ['date', 'data', 'time', 'datetime'], required=True)
    df = df.rename(columns={date_col: 'date'})
    df['date'] = pd.to_datetime(df['date'])

    flow_col = column_mapping.get('flow')
    if flow_col is None:
        flow_col = _find_column(df, ['streamflow_m3s', 'flow', 'vazao', 'q', 'mean_Q', 'discharge'], required=False)
        if flow_col is None:
            raise ValueError(f"Estação {station_id}: Coluna de vazão não encontrada")

    if flow_col in df.columns:
        df = df.rename(columns={flow_col: 'streamflow_m3s'})
    else:
        raise ValueError(f"Estação {station_id}: Coluna '{flow_col}' não encontrada no arquivo")

    precip_col = column_mapping.get('precip')
    has_precip = False
    if precip_col is None:
        precip_col = _find_column(df, ['precipitation_chirps', 'precipitation', 'precip', 'mean_P', 'P'], required=False)

    if precip_col and precip_col in df.columns:
        df = df.rename(columns={precip_col: 'precipitation_chirps'})
        has_precip = True
    elif verbose:
        print(f"      ⚠️ Coluna de precipitação não encontrada")

    et_col = column_mapping.get('et')
    has_et = False
    if et_col is not None and et_col in df.columns:
        df = df.rename(columns={et_col: 'potential_evapotransp_gleam'})
        has_et = True

    required_cols = ['date', 'streamflow_m3s']
    if has_precip:
        required_cols.append('precipitation_chirps')
    if has_et:
        required_cols.append('potential_evapotransp_gleam')

    existing_cols = [c for c in required_cols if c in df.columns]
    df = df[existing_cols]
    df = df.sort_values('date').reset_index(drop=True)
    return df


def _load_forecast_precipitation(data_dir, station_ids, column_mapping, verbose=True):
    forecast_dict = {}
    for station_id in station_ids:
        possible_files = [
            data_dir / f"{station_id}_precipitation_forecast.csv",
            data_dir / f"{station_id}_forecast.csv",
        ]
        file_path = None
        for pf in possible_files:
            if pf.exists():
                file_path = pf
                break
        if file_path is None:
            if verbose:
                print(f"   ⚠️ Sem forecast de precipitação para estação {station_id}")
            continue
        try:
            df = pd.read_csv(file_path)
            date_col = _find_column(df, ['date', 'data'], required=True)
            df['date'] = pd.to_datetime(df[date_col])

            precip_col = None
            if column_mapping.get('precip') is not None:
                if column_mapping['precip'] in df.columns:
                    precip_col = column_mapping['precip']
            if precip_col is None:
                precip_col = _find_column(df, ['precipitation_forecast', 'precipitation', 'precip', 'mean_P', 'P'], required=False)

            if precip_col:
                df = df[['date', precip_col]].rename(columns={precip_col: 'precipitation_forecast'})
            else:
                if verbose:
                    print(f"   ⚠️ Coluna de precipitação não encontrada em {station_id}")
                continue

            forecast_dict[station_id] = df
            if verbose:
                print(f"   ✅ Forecast {station_id}: {len(df)} registros ({df['date'].min().date()} até {df['date'].max().date()})")
        except Exception as e:
            if verbose:
                print(f"   ❌ Erro forecast {station_id}: {e}")
            continue
    return forecast_dict


def _create_continuous_series_with_reference(observed_dict, forecast_dict, station_ids, reference_dates, verbose=True):
    merged_dict = {}
    ref_dates = pd.to_datetime(reference_dates)
    max_ref_date = ref_dates.max()

    for station_id in station_ids:
        if station_id not in observed_dict:
            continue
        df_obs = observed_dict[station_id].copy()

        if verbose:
            print(f"   📊 Estação {station_id}:")
            print(f"      Observados: {df_obs['date'].min().date()} até {df_obs['date'].max().date()}")
            print(f"      Reference date: {max_ref_date.date()}")

        if station_id in forecast_dict:
            df_fc = forecast_dict[station_id].copy()
            if verbose:
                print(f"      Forecast: {df_fc['date'].min().date()} até {df_fc['date'].max().date()}")

            df_merged = df_obs.merge(df_fc, on='date', how='outer', suffixes=('_obs', '_fc'))
            df_merged = df_merged.sort_values('date').reset_index(drop=True)

            # OBS: ATÉ ref_date = dados, DEPOIS = NaN
            if 'precipitation_chirps' in df_merged.columns:
                mask_after_ref = df_merged['date'] > max_ref_date
                df_merged.loc[mask_after_ref, 'precipitation_chirps'] = np.nan

            # Forecast: ATÉ ref_date = NaN, DEPOIS = dados
            if 'precipitation_forecast' in df_merged.columns:
                mask_before_or_equal_ref = df_merged['date'] <= max_ref_date
                df_merged.loc[mask_before_or_equal_ref, 'precipitation_forecast'] = np.nan

            if verbose:
                obs_count = df_merged['precipitation_chirps'].notna().sum() if 'precipitation_chirps' in df_merged.columns else 0
                fc_count = df_merged['precipitation_forecast'].notna().sum() if 'precipitation_forecast' in df_merged.columns else 0
                print(f"      ✅ Resultado: {obs_count} dias observados (até {max_ref_date.date()}), {fc_count} dias forecast (após {max_ref_date.date()})")
        else:
            df_merged = df_obs.copy()
            df_merged['precipitation_forecast'] = np.nan
            mask_after_ref = df_merged['date'] > max_ref_date
            if 'precipitation_chirps' in df_merged.columns:
                df_merged.loc[mask_after_ref, 'precipitation_chirps'] = np.nan
            if verbose:
                print("      ⚠️ Sem forecast - apenas observados até reference_date")

        if 'streamflow_m3s' not in df_merged.columns:
            raise ValueError(f"Estação {station_id}: coluna streamflow_m3s não encontrada")

        # Streamflow: ATÉ reference_date
        if 'streamflow_m3s' in df_merged.columns:
            mask_after_ref = df_merged['date'] > max_ref_date
            df_merged.loc[mask_after_ref, 'streamflow_m3s'] = np.nan

        cols_to_keep = ['date', 'streamflow_m3s']
        if 'precipitation_chirps' in df_merged.columns:
            cols_to_keep.append('precipitation_chirps')
        if 'precipitation_forecast' in df_merged.columns:
            cols_to_keep.append('precipitation_forecast')
        if 'potential_evapotransp_gleam' in df_merged.columns:
            cols_to_keep.append('potential_evapotransp_gleam')

        df_merged = df_merged[cols_to_keep]
        merged_dict[station_id] = df_merged

    return merged_dict


def _generate_features(merged_dict, config, verbose=True):
    forcings = config.forcings
    reference_dates = config.reference_dates
    max_ref_date = pd.to_datetime(reference_dates).max()

    engineer = HydroFeatureEngineer(
        api_k_list=config.api_k_list or [0.85, 0.95, 0.98],
        precipitation_ma_windows=config.precipitation_ma_windows or [3, 5, 7, 10],
        precipitation_cumulative_windows=config.precipitation_cumulative_windows or [3, 5, 7, 10],
        forecast_ma_windows=config.forecast_ma_windows or [3, 5, 7],
        forecast_cumulative_windows=config.forecast_cumulative_windows or [3, 5, 7],
        evapotranspiration_ma_windows=config.evapotranspiration_ma_windows or [3, 5, 7],
        anomaly_ma_windows=config.anomaly_ma_windows or [7, 15, 30],
        forcings=forcings
    )

    if not merged_dict:
        raise ValueError("merged_dict vazio - sem dados para processar")

    sample_df = list(merged_dict.values())[0]
    available_cols = set(sample_df.columns)

    if verbose:
        print(f"   📋 Forçantes: {forcings}")
        print(f"   📋 Reference date: {max_ref_date.date()}")
        print(f"   📋 Colunas disponíveis: {available_cols}")

    column_names = {'flow': 'streamflow_m3s'}

    if 'precipitation_chirps' in available_cols:
        column_names['precip_obs'] = 'precipitation_chirps'

    if 'precipitation_forecast' in available_cols:
        column_names['precip_forecast'] = 'precipitation_forecast'

    if forcings == "P_ET" and 'potential_evapotransp_gleam' in available_cols:
        column_names['et_obs'] = 'potential_evapotransp_gleam'
        column_names['et_forecast'] = 'potential_evapotransp_gleam'

    if verbose:
        print(f"   📋 column_names: {column_names}")

    df_features = engineer.process_multiple_stations(
        merged_dict,
        train_date_cutoff=None,
        column_names=column_names,
        forcings=forcings
    )

    if verbose:
        print(f"   🔒 Aplicando mascaramento basado em reference_date = {max_ref_date.date()}...")

    df_features = _apply_forecast_mask(df_features, max_ref_date, verbose=verbose)

    return df_features


def _apply_forecast_mask(df_features, reference_date, verbose=True):
    """
    Aplica mascaramento final baseado no reference_date.
    
    REGRA:
    - Colunas COM "_forecast" no nome: dados A PARTIR DE reference_date (APOS)
    - Colunas SEM "_forecast" (OBS): dados ATE reference_date (INCLUINDO)
    """
    df = df_features.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    mask_future = df.index > reference_date
    mask_past = df.index <= reference_date
    
    cols_masked_future = 0
    cols_masked_past = 0
    cols_first_fixed = 0
    
    for col in df.columns:
        col_lower = col.lower()
        has_forecast_suffix = '_forecast' in col_lower
        
        if has_forecast_suffix:
            # FORECAST: dados validos APOS reference_date
            old_count = df.loc[mask_past, col].notna().sum()
            if old_count > 0:
                df.loc[mask_past, col] = np.nan
                cols_masked_past += 1
        else:
            # OBS: dados validos ATE reference_date
            old_count = df.loc[mask_future, col].notna().sum()
            if old_count > 0:
                df.loc[mask_future, col] = np.nan
                cols_masked_future += 1
            
            # dP/dt: primeiro valor = 0 deve ser NaN
            if 'dp' in col_lower and 'dt' in col_lower:
                if len(df) > 0:
                    first_val = df.iloc[0][col]
                    if pd.notna(first_val) and first_val == 0:
                        df.iloc[0, df.columns.get_loc(col)] = np.nan
                        cols_first_fixed += 1
    
    if verbose:
        print(f"   -> {cols_masked_future} colunas OBS com dados futurados mascarados (agora NaN)")
        print(f"   -> {cols_masked_past} colunas FORECAST com dados passados mascarados (agora NaN)")
        print(f"   -> {cols_first_fixed} colunas dP/dt com primeiro valor corrigido para NaN")
    
    return df


def _find_column(df, possible_names, required=False):
    for name in possible_names:
        if name in df.columns:
            return name
        for col in df.columns:
            if col.lower() == name.lower():
                return col
    if required:
        available = list(df.columns)
        raise ValueError(f"Coluna não encontrada. Nomes tentados: {possible_names}. Disponíveis: {available}")
    return None
