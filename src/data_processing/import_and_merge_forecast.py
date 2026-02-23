"""
Módulo para importar e fundir dados de forecast com dados observados.
"""

from typing import Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

def load_forecast_data(
    forecast_dir: Path,
    station_id: int
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Carrega arquivos de forecast de precipitação e evapotranspiração para uma estação.

    Args:
        forecast_dir: Diretório contendo os arquivos de forecast
        station_id: ID da estação

    Returns:
        Tuple (df_precip, df_et) ou (None, None) se arquivos não existirem
    """
    precip_path = forecast_dir / f"{station_id}_precipitation_forecast.csv"
    et_path = forecast_dir / f"{station_id}_evapotranspiration_forecast.csv"

    if not precip_path.exists() or not et_path.exists():
        print(f"⚠️ Avisos: Arquivos de forecast não encontrados para estação {station_id}")
        return None, None

    df_precip = pd.read_csv(precip_path, parse_dates=['date'])
    df_et = pd.read_csv(et_path, parse_dates=['date'])

    return df_precip, df_et

def merge_forecast_with_observed(
    df_observed: pd.DataFrame,
    df_precip_forecast: pd.DataFrame,
    df_et_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """
    Funde dados observados com dados de forecast.

    Estratégia:
    1. Mantém todos os dados observados originais
    2. Adiciona colunas de forecast (precipitation_forecast, et_forecast)
    3. Para o período observado, forecast = observado (ou zero se não houver)
    4. Para o período futuro (forecast), preenche com os dados dos arquivos de forecast
    5. Estende o DataFrame para incluir as datas futuras
    """
    df = df_observed.copy()

    # Garantir que datas são datetime
    df['date'] = pd.to_datetime(df['date'])
    df_precip_forecast['date'] = pd.to_datetime(df_precip_forecast['date'])
    df_et_forecast['date'] = pd.to_datetime(df_et_forecast['date'])

    last_obs_date = df['date'].max()

    # Filtrar forecast para apenas datas APÓS o último dado observado
    # (para evitar duplicatas ou conflitos, assumimos que forecast estende a série)
    df_precip_future = df_precip_forecast[df_precip_forecast['date'] > last_obs_date].copy()
    df_et_future = df_et_forecast[df_et_forecast['date'] > last_obs_date].copy()

    # Se não houver dados futuros, retorna o original com colunas vazias/preenchidas
    if df_precip_future.empty:
        return df

    # As colunas que não existem no forecast ficarão como NaN (ex: vazão Q)

    # Merge dos forecasts futuros
    df_future = pd.merge(df_precip_future, df_et_future, on=['date', 'station_id'], how='outer')

    # Renomear colunas para bater com o esperado se necessário,

    # No DF observado, precisamos criar essas colunas de forecast se não existirem

    # Vamos criar as colunas de forecast no DF observado
    if 'precipitation_forecast' not in df.columns:
        # Se não existe, usamos a precipitação observada como "forecast passado"
        if 'precipitation_chirps' in df.columns:
            df['precipitation_forecast'] = df['precipitation_chirps']
        else:
            df['precipitation_forecast'] = np.nan

    if 'et_forecast' not in df.columns:
        if 'potential_evapotransp_gleam' in df.columns:
            df['et_forecast'] = df['potential_evapotransp_gleam']
        else:
            df['et_forecast'] = np.nan

    # Agora concatenamos
    # O df_future tem 'date', 'station_id', 'precipitation_forecast', 'et_forecast'
    # O df_observed tem muitas outras colunas.

    df_combined = pd.concat([df, df_future], ignore_index=True)
    df_combined = df_combined.sort_values('date').reset_index(drop=True)

    # Preencher NaNs nas colunas de forecast do futuro (já devem estar ok)
    # Preencher NaNs nas colunas observadas do futuro (Q, P_obs, ET_obs) -> Devem ficar NaN mesmo
    # pois não temos observação no futuro.

    return df_combined
