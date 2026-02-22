"""
Módulo para feature engineering de dados hidrológicos.
"""

from typing import Dict, List, Optional
import pathlib
import sys
import pandas as pd
import numpy as np
import yaml

# Corrigir a importação do config_loader
current_dir = pathlib.Path(__file__).parent
project_root = current_dir.parent.parent

# Adicionar o src ao sys.path
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from utils.config_loader import load_feature_config
    CONFIG_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    CONFIG_AVAILABLE = False
    print(f"⚠️  Módulo config_loader não encontrado: {e}. Usando valores padrão.")


class HydroFeatureEngineer:
    """
    Classe para criação de features hidrológicas avançadas.
    """

    def __init__(self,
                 api_k_list: List[float] = None,
                 precipitation_ma_windows: List[int] = None,
                 precipitation_cumulative_windows: List[int] = None,
                 forecast_ma_windows: List[int] = None,
                 forecast_cumulative_windows: List[int] = None,
                 evapotranspiration_ma_windows: List[int] = None,
                 anomaly_ma_windows: List[int] = None):
        """
        Inicializa o engenheiro de features com configurações granulares.

        Args:
            api_k_list: Lista de valores de k para API
            precipitation_ma_windows: Janelas para médias móveis de precipitação observada
            precipitation_cumulative_windows: Janelas para acumulados de precipitação observada
            forecast_ma_windows: Janelas para médias móveis de forecast de precipitação
            forecast_cumulative_windows: Janelas para acumulados de forecast de precipitação
            evapotranspiration_ma_windows: Janelas para médias móveis de evapotranspiração
            anomaly_ma_windows: Janelas para médias móveis de anomalias
        """
        # Valores padrão
        self.api_k_list = api_k_list or [0.70, 0.80, 0.85, 0.90, 0.92, 0.95]
        self.precipitation_ma_windows = precipitation_ma_windows or [3, 7, 15]
        self.precipitation_cumulative_windows = precipitation_cumulative_windows or [3, 5, 7, 10]
        self.forecast_ma_windows = forecast_ma_windows or [3, 7, 15]
        self.forecast_cumulative_windows = forecast_cumulative_windows or [3, 5, 7, 10]
        self.evapotranspiration_ma_windows = evapotranspiration_ma_windows or [7, 14, 30]
        self.anomaly_ma_windows = anomaly_ma_windows or [3, 7]

    @staticmethod
    def compute_api(series: pd.Series, k: float) -> pd.Series:
        """
        Calcula o Antecedent Precipitation Index (API).
        """
        vals = series.to_numpy(dtype=np.float64)
        api = np.zeros_like(vals)

        for i in range(len(vals)):
            p = vals[i] if not np.isnan(vals[i]) else 0.0
            api[i] = p if i == 0 else (p + k * api[i - 1])

        return pd.Series(api, index=series.index)

    def add_precipitation_features(self,
                                  df: pd.DataFrame,
                                  station: int,
                                  precip_col: str,
                                  is_forecast: bool = False) -> pd.DataFrame:
        """
        Adiciona features de precipitação (observada ou forecast).
        """
        if is_forecast:
            ma_windows = self.forecast_ma_windows
            cum_windows = self.forecast_cumulative_windows
            prefix = "precipitation_forecast"
        else:
            ma_windows = self.precipitation_ma_windows
            cum_windows = self.precipitation_cumulative_windows
            prefix = "precipitation_obs"

        # Médias Móveis
        for w in ma_windows:
            min_p = max(1, w // 2)
            col_name = f'{prefix}_ma{w}_{station}'
            df[col_name] = df[precip_col].rolling(window=w, min_periods=min_p).mean()

        # Acumulados
        for w in cum_windows:
            min_p = max(1, w // 2)
            col_name = f'{prefix}_cum{w}_{station}'
            df[col_name] = df[precip_col].rolling(window=w, min_periods=min_p).sum()

        return df

    def add_evapotranspiration_features(self,
                                       df: pd.DataFrame,
                                       station: int,
                                       et_col: str,
                                       is_forecast: bool = False) -> pd.DataFrame:
        """
        Adiciona features de evapotranspiração (observada ou forecast).
        """
        prefix = "et_forecast" if is_forecast else "et_obs"
        
        for w in self.evapotranspiration_ma_windows:
            min_p = max(1, w // 2)
            df[f'{prefix}_ma{w}_{station}'] = (
                df[et_col].rolling(window=w, min_periods=min_p).mean()
            )

        return df

    def add_api_features(self,
                        df: pd.DataFrame,
                        station: int,
                        precip_obs_col: str,
                        precip_forecast_col: str) -> pd.DataFrame:
        """
        Adiciona features de API para dados observados e forecast.
        """
        for k in self.api_k_list:
            tag = f"k{int(round(k*100)):02d}"
            df[f'api_obs_{tag}_{station}'] = self.compute_api(df[precip_obs_col], k)
            df[f'api_forecast_{tag}_{station}'] = self.compute_api(df[precip_forecast_col], k)

        return df

    def add_anomaly_features(self,
                            df: pd.DataFrame,
                            station: int,
                            anomaly_prefix: str) -> pd.DataFrame:
        """
        Adiciona features de anomalia.
        """
        full_anomaly_col = f'{anomaly_prefix}_{station}'

        for w in self.anomaly_ma_windows:
            min_p = max(1, w // 2)
            df[f'{anomaly_prefix}_ma{w}_{station}'] = (
                df[full_anomaly_col].rolling(window=w, min_periods=min_p)
                .mean()
                .shift(1)
                .fillna(0.0)
            )

        return df

    def add_advanced_features(self,
                             df: pd.DataFrame,
                             station: int,
                             flow_col: str,
                             precip_obs_col: str,
                             precip_forecast_col: str,
                             train_date_cutoff: Optional[str] = None) -> pd.DataFrame:
        """
        Adiciona features avançadas.
        """
        # Definir período de referência
        if train_date_cutoff is not None:
            ref_mask = df.index <= pd.to_datetime(train_date_cutoff)
            ref_df = df.loc[ref_mask]
        else:
            ref_df = df

        # Mediana sazonal
        seasonal_median = ref_df.groupby(ref_df.index.dayofyear)[flow_col].median()

        # Derivadas
        df[f'dQ_dt_{station}'] = df[flow_col].diff().fillna(0.0)
        df[f'dP_obs_dt_{station}'] = df[precip_obs_col].diff().fillna(0.0)
        df[f'dP_forecast_dt_{station}'] = df[precip_forecast_col].diff().fillna(0.0)

        # Estado do regime
        dQ_smooth = df[f'dQ_dt_{station}'].rolling(5, center=True, min_periods=1).mean()
        dQ_std_norm = (dQ_smooth - dQ_smooth.mean()) / (dQ_smooth.std() + 1e-6)

        regime_state = np.zeros(len(dQ_std_norm), dtype=np.int8)
        state = 0
        counter = 0

        vals_std = dQ_std_norm.to_numpy()
        for i in range(1, len(vals_std)):
            val = vals_std[i]
            counter += 1

            if state <= 0 and val >= 0.2:
                counter = 0
                state = 1
            elif state >= 0 and val <= -0.25:
                counter = 0
                state = -1
            elif abs(val) < 0.1 and counter >= 3:
                state = 0
                counter = 0

            regime_state[i] = state

        df[f'regime_state_{station}'] = regime_state

        # Anomalia logarítmica
        doy_series = df.index.dayofyear
        median_vals = doy_series.map(seasonal_median).fillna(0.0).to_numpy()

        log_q = np.log1p(df[flow_col].to_numpy())
        log_median = np.log1p(median_vals)

        df[f'log_anomaly_{station}'] = log_q - log_median
        df = self.add_anomaly_features(df, station, 'log_anomaly')

        return df

    def add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features sazonais.
        """
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12.0)

        return df

    def process_station(self,
                       df: pd.DataFrame,
                       station_id: int,
                       train_date_cutoff: Optional[str] = None) -> pd.DataFrame:
        """
        Processa uma única estação, adicionando todas as features.
        
        IMPORTANTE: O DataFrame de entrada deve conter as colunas de forecast
        já merged de data/forecast/
        """
        # Verificar colunas necessárias
        required_cols = [
            'date', 'streamflow_m3s',
            'precipitation_chirps', 'potential_evapotransp_gleam',
            'precipitation_forecast', 'et_forecast'  # ← Devem vir do merge!
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"❌ Colunas faltantes para estação {station_id}: {missing_cols}\n"
                f"   Certifique-se de fazer o merge com dados de data/forecast/ "
                f"antes de processar features."
            )

        # Preparar DataFrame
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # Renomear colunas com padronização _obs e _forecast
        column_map = {
            'streamflow_m3s': f'Q_{station_id}',
            'precipitation_chirps': f'precipitation_obs_{station_id}',
            'potential_evapotransp_gleam': f'et_obs_{station_id}',
            'precipitation_forecast': f'precipitation_forecast_{station_id}',
            'et_forecast': f'et_forecast_{station_id}'
        }

        df = df.rename(columns=column_map)

        # Definir nomes das colunas
        flow_col = f'Q_{station_id}'
        precip_obs_col = f'precipitation_obs_{station_id}'
        precip_fc_col = f'precipitation_forecast_{station_id}'
        et_obs_col = f'et_obs_{station_id}'
        et_fc_col = f'et_forecast_{station_id}'

        # ==========================================
        # FEATURES DE PRECIPITAÇÃO
        # ==========================================
        # Observada (para encoder)
        df = self.add_precipitation_features(df, station_id, precip_obs_col, is_forecast=False)
        
        # Forecast (para decoder)
        df = self.add_precipitation_features(df, station_id, precip_fc_col, is_forecast=True)

        # ==========================================
        # FEATURES DE EVAPOTRANSPIRAÇÃO
        # ==========================================
        # Observada (para encoder)
        df = self.add_evapotranspiration_features(df, station_id, et_obs_col, is_forecast=False)
        
        # Forecast (para decoder)
        df = self.add_evapotranspiration_features(df, station_id, et_fc_col, is_forecast=True)

        # ==========================================
        # API
        # ==========================================
        df = self.add_api_features(df, station_id, precip_obs_col, precip_fc_col)

        # ==========================================
        # FEATURES AVANÇADAS
        # ==========================================
        df = self.add_advanced_features(
            df, station_id, flow_col, precip_obs_col, precip_fc_col, train_date_cutoff
        )

        return df

    def process_multiple_stations(self,
                                 data_dict: Dict[int, pd.DataFrame],
                                 train_date_cutoff: Optional[str] = None) -> pd.DataFrame:
        """
        Processa múltiplas estações e combina em um único DataFrame.
        """
        processed_dfs = []

        for station_id, df in data_dict.items():
            try:
                df_processed = self.process_station(df, station_id, train_date_cutoff)
                processed_dfs.append(df_processed)
                print(f"✓ Estação {station_id} processada")
            except Exception as e:
                print(f"✗ Erro ao processar estação {station_id}: {e}")
                continue

        if not processed_dfs:
            raise ValueError("Nenhuma estação foi processada com sucesso")

        # Combinar DataFrames
        combined_df = processed_dfs[0]
        for df in processed_dfs[1:]:
            combined_df = combined_df.merge(df, left_index=True, right_index=True, how='outer')

        # Preencher valores faltantes
        combined_df = combined_df.ffill().bfill()

        # Adicionar features sazonais
        combined_df = self.add_seasonal_features(combined_df)

        return combined_df


def load_station_data(complete_series_dir: pathlib.Path,
                     station_ids: List[int]) -> Dict[int, pd.DataFrame]:
    """
    Carrega dados de múltiplas estações da pasta complete_series.
    """
    data_dict = {}

    for station_id in station_ids:
        file_path = complete_series_dir / f"{station_id}_complete_date.csv"

        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                data_dict[station_id] = df
            except Exception as e:
                print(f"✗ Erro ao carregar estação {station_id}: {e}")
        else:
            print(f"✗ Arquivo não encontrado: {file_path}")

    return data_dict


def load_forecast_data(forecast_dir: pathlib.Path,
                      station_ids: List[int]) -> Dict[int, pd.DataFrame]:
    """
    Carrega dados de forecast para múltiplas estações.
    
    Returns:
        Dicionário {station_id: DataFrame com colunas 'precipitation_forecast' e 'et_forecast'}
    """
    forecast_dict = {}

    for station_id in station_ids:
        precip_file = forecast_dir / f"{station_id}_precipitation_forecast.csv"
        et_file = forecast_dir / f"{station_id}_evapotranspiration_forecast.csv"

        if not precip_file.exists() or not et_file.exists():
            print(f"✗ Arquivos de forecast não encontrados para estação {station_id}")
            continue

        try:
            # Carregar precipitação
            df_precip = pd.read_csv(precip_file, parse_dates=['date'])
            df_precip = df_precip[['date', 'precipitation_forecast']]

            # Carregar ET
            df_et = pd.read_csv(et_file, parse_dates=['date'])
            df_et = df_et[['date', 'et_forecast']]

            # Merge
            df_forecast = df_precip.merge(df_et, on='date', how='outer')
            forecast_dict[station_id] = df_forecast

        except Exception as e:
            print(f"✗ Erro ao carregar forecast da estação {station_id}: {e}")

    return forecast_dict


def merge_observed_and_forecast(
    observed_dict: Dict[int, pd.DataFrame],
    forecast_dict: Dict[int, pd.DataFrame]
) -> Dict[int, pd.DataFrame]:
    """
    Faz merge dos dados observados com forecast.
    
    Returns:
        Dicionário {station_id: DataFrame combinado}
    """
    merged_dict = {}

    for station_id in observed_dict.keys():
        if station_id not in forecast_dict:
            print(f"⚠️  Sem dados de forecast para estação {station_id}, pulando...")
            continue

        df_obs = observed_dict[station_id].copy()
        df_fc = forecast_dict[station_id].copy()

        # Merge por data
        df_merged = df_obs.merge(df_fc, on='date', how='left')

        # Para dias sem forecast, usar dados observados
        df_merged['precipitation_forecast'] = df_merged['precipitation_forecast'].fillna(
            df_merged['precipitation_chirps']
        )
        df_merged['et_forecast'] = df_merged['et_forecast'].fillna(
            df_merged['potential_evapotransp_gleam']
        )

        merged_dict[station_id] = df_merged

    return merged_dict


def process_features(input_dir: pathlib.Path,
                    forecast_dir: pathlib.Path,
                    output_dir: pathlib.Path,
                    station_ids: List[int],
                    api_k_list: Optional[List[float]] = None,
                    precipitation_ma_windows: Optional[List[int]] = None,
                    precipitation_cumulative_windows: Optional[List[int]] = None,
                    forecast_ma_windows: Optional[List[int]] = None,
                    forecast_cumulative_windows: Optional[List[int]] = None,
                    evapotranspiration_ma_windows: Optional[List[int]] = None,
                    anomaly_ma_windows: Optional[List[int]] = None,
                    train_date_cutoff: Optional[str] = None,
                    output_filename: str = "features_combined.csv",
                    use_config_file: bool = True) -> pd.DataFrame:
    """
    Função principal para processamento de features.
    Agora inclui merge com dados de forecast.
    """
    print("="*60)
    print("PROCESSAMENTO DE FEATURES")
    print("="*60)

    # Carregar configurações do YAML se disponível
    if use_config_file and CONFIG_AVAILABLE:
        try:
            config = load_feature_config()
            print("✅ Configurações carregadas do arquivo YAML")

            if api_k_list is None:
                api_k_list = config.get('api_k_list')
            if precipitation_ma_windows is None:
                precipitation_ma_windows = config.get('precipitation_ma')
            if precipitation_cumulative_windows is None:
                precipitation_cumulative_windows = config.get('precipitation_cum')
            if forecast_ma_windows is None:
                forecast_ma_windows = config.get('forecast_ma', config.get('precipitation_ma'))
            if forecast_cumulative_windows is None:
                forecast_cumulative_windows = config.get('forecast_cum', config.get('precipitation_cum'))
            if evapotranspiration_ma_windows is None:
                evapotranspiration_ma_windows = config.get('evapotranspiration_ma')
            if anomaly_ma_windows is None:
                anomaly_ma_windows = config.get('anomaly_ma')

        except Exception as e:
            print(f"⚠️  Erro ao carregar configurações: {e}")

    # Verificar diretórios
    if not input_dir.exists():
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_dir}")
    if not forecast_dir.exists():
        raise FileNotFoundError(f"Diretório de forecast não encontrado: {forecast_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Carregar dados observados
    print(f"\n📊 Carregando dados observados de {len(station_ids)} estações...")
    observed_dict = load_station_data(input_dir, station_ids)
    print(f"✓ {len(observed_dict)} estações carregadas")

    # Carregar dados de forecast
    print("\n🔮 Carregando dados de forecast...")
    forecast_dict = load_forecast_data(forecast_dir, station_ids)
    print(f"✓ {len(forecast_dict)} estações com forecast carregadas")

    # Merge observado + forecast
    print("\n🔗 Fazendo merge observado + forecast...")
    merged_dict = merge_observed_and_forecast(observed_dict, forecast_dict)
    print(f"✓ {len(merged_dict)} estações combinadas")

    if not merged_dict:
        raise ValueError("Nenhum dado foi combinado. Verifique os arquivos de forecast.")

    # Processar features
    print("\n⚙️  Criando features...")
    engineer = HydroFeatureEngineer(
        api_k_list=api_k_list,
        precipitation_ma_windows=precipitation_ma_windows,
        precipitation_cumulative_windows=precipitation_cumulative_windows,
        forecast_ma_windows=forecast_ma_windows,
        forecast_cumulative_windows=forecast_cumulative_windows,
        evapotranspiration_ma_windows=evapotranspiration_ma_windows,
        anomaly_ma_windows=anomaly_ma_windows
    )

    combined_df = engineer.process_multiple_stations(
        merged_dict,
        train_date_cutoff=train_date_cutoff
    )

    # Salvar resultados
    output_path = output_dir / output_filename
    combined_df.to_csv(output_path)

    print("\n" + "="*60)
    print("✅ FEATURES CRIADAS COM SUCESSO")
    print("="*60)
    print(f"  Estações processadas: {len(merged_dict)}")
    print(f"  Período: {combined_df.index.min().date()} a {combined_df.index.max().date()}")
    print(f"  Total de dias: {len(combined_df)}")
    print(f"  Total de features: {len(combined_df.columns)}")
    print(f"  Arquivo salvo: {output_path}")
    print("="*60)

    return combined_df
