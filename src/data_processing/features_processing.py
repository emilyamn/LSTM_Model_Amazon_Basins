"""
Módulo para feature engineering de dados hidrológicos.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import pathlib


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
            api_k_list: Lista de valores de k para API (Antecedent Precipitation Index)
            precipitation_ma_windows: Janelas para médias móveis de precipitação observada
            precipitation_cumulative_windows: Janelas para acumulados de precipitação observada
            forecast_ma_windows: Janelas para médias móveis de forecast de precipitação
            forecast_cumulative_windows: Janelas para acumulados de forecast de precipitação
            evapotranspiration_ma_windows: Janelas para médias móveis de evapotranspiração
            anomaly_ma_windows: Janelas para médias móveis de anomalias
        """
        # Valores padrão com separação por tipo de feature
        self.api_k_list = api_k_list or [0.70, 0.80, 0.85, 0.90, 0.92, 0.95]
        
        # Precipitação observada
        self.precipitation_ma_windows = precipitation_ma_windows or [3, 7, 15]
        self.precipitation_cumulative_windows = precipitation_cumulative_windows or [3, 5, 7, 10]
        
        # Forecast de precipitação (pode ser diferente)
        self.forecast_ma_windows = forecast_ma_windows or [3, 7, 15]
        self.forecast_cumulative_windows = forecast_cumulative_windows or [3, 5, 7, 10]
        
        # Evapotranspiração
        self.evapotranspiration_ma_windows = evapotranspiration_ma_windows or [7, 14, 30]
        
        # Anomalias
        self.anomaly_ma_windows = anomaly_ma_windows or [3, 7]
    
    @staticmethod
    def compute_api(series: pd.Series, k: float) -> pd.Series:
        """
        Calcula o Antecedent Precipitation Index (API).
        
        Args:
            series: Série temporal de precipitação
            k: Fator de decaimento (0 < k < 1)
            
        Returns:
            Série com valores de API
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
        
        Args:
            df: DataFrame com dados
            station: ID da estação
            precip_col: Nome da coluna de precipitação
            is_forecast: Se True, usa configurações de forecast
            
        Returns:
            DataFrame com features adicionadas
        """
        if is_forecast:
            ma_windows = self.forecast_ma_windows
            cum_windows = self.forecast_cumulative_windows
            prefix = "precipitation_forecast"
        else:
            ma_windows = self.precipitation_ma_windows
            cum_windows = self.precipitation_cumulative_windows
            prefix = "precipitation"
        
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
                                       et_col: str) -> pd.DataFrame:
        """
        Adiciona features de evapotranspiração.
        
        Args:
            df: DataFrame com dados
            station: ID da estação
            et_col: Nome da coluna de evapotranspiração
            
        Returns:
            DataFrame com features adicionadas
        """
        for w in self.evapotranspiration_ma_windows:
            min_p = max(1, w // 2)
            df[f'evapotranspiration_ma{w}_{station}'] = (
                df[et_col].rolling(window=w, min_periods=min_p).mean()
            )
        
        return df
    
    def add_api_features(self,
                        df: pd.DataFrame,
                        station: int,
                        precip_col: str,
                        forecast_col: str) -> pd.DataFrame:
        """
        Adiciona features de API (Antecedent Precipitation Index).
        
        Args:
            df: DataFrame com dados
            station: ID da estação
            precip_col: Nome da coluna de precipitação observada
            forecast_col: Nome da coluna de previsão de precipitação
            
        Returns:
            DataFrame com features de API adicionadas
        """
        for k in self.api_k_list:
            tag = f"k{int(round(k*100)):02d}"
            df[f'api_{tag}_{station}'] = self.compute_api(df[precip_col], k)
            df[f'api_forecast_{tag}_{station}'] = self.compute_api(df[forecast_col], k)
        
        return df
    
    def add_anomaly_features(self,
                            df: pd.DataFrame,
                            station: int,
                            anomaly_prefix: str) -> pd.DataFrame:
        """
        Adiciona features de anomalia.
        
        Args:
            df: DataFrame com dados
            station: ID da estação
            anomaly_prefix: Prefixo da coluna de anomalia (ex: 'log_anomaly')
            
        Returns:
            DataFrame com features de anomalia adicionadas
        """
        # Nome completo da coluna de anomalia
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
                             precip_col: str,
                             forecast_col: str,
                             train_date_cutoff: Optional[str] = None) -> pd.DataFrame:
        """
        Adiciona features avançadas.
        
        Args:
            df: DataFrame com dados
            station: ID da estação
            flow_col: Nome da coluna de vazão
            precip_col: Nome da coluna de precipitação
            forecast_col: Nome da coluna de previsão
            train_date_cutoff: Data de corte para cálculo de estatísticas
            
        Returns:
            DataFrame com features avançadas
        """
        # Definir período de referência para estatísticas
        if train_date_cutoff is not None:
            ref_mask = df.index <= pd.to_datetime(train_date_cutoff)
            ref_df = df.loc[ref_mask]
        else:
            ref_df = df
        
        # Calcular mediana sazonal
        seasonal_median = ref_df.groupby(ref_df.index.dayofyear)[flow_col].median()
        
        # Derivadas
        df[f'dQ_dt_{station}'] = df[flow_col].diff().fillna(0.0)
        df[f'dP_dt_{station}'] = df[precip_col].diff().fillna(0.0)
        df[f'dP_dt_forecast_{station}'] = df[forecast_col].diff().fillna(0.0)
        
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
        
        # Adicionar médias móveis da anomalia
        df = self.add_anomaly_features(df, station, 'log_anomaly')
        
        return df
    
    def add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features sazonais.
        
        Args:
            df: DataFrame com dados
            
        Returns:
            DataFrame com features sazonais adicionadas
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
        
        Args:
            df: DataFrame original da estação
            station_id: ID da estação
            train_date_cutoff: Data de corte para estatísticas
            
        Returns:
            DataFrame com features
        """
        # Verificar colunas necessárias
        required_cols = ['date', 'streamflow_m3s', 'precipitation_chirps', 
                        'potential_evapotransp_gleam']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colunas faltantes para estação {station_id}: {missing_cols}")
        
        # Preparar DataFrame
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Renomear colunas
        column_map = {
            'streamflow_m3s': f'Q_{station_id}',
            'precipitation_chirps': f'precipitation_{station_id}',
            'potential_evapotransp_gleam': f'evapotranspiration_{station_id}'
        }
        
        df = df.rename(columns=column_map)
        
        # Criar colunas de forecast (mesmos dados para início)
        df[f'precipitation_forecast_{station_id}'] = df[f'precipitation_{station_id}'].copy()
        df[f'evapotranspiration_forecast_{station_id}'] = df[f'evapotranspiration_{station_id}'].copy()
        
        # Adicionar features específicas
        precip_col = f'precipitation_{station_id}'
        precip_fc_col = f'precipitation_forecast_{station_id}'
        et_col = f'evapotranspiration_{station_id}'
        flow_col = f'Q_{station_id}'
        
        # Precipitação observada
        df = self.add_precipitation_features(df, station_id, precip_col, is_forecast=False)
        
        # Forecast de precipitação
        df = self.add_precipitation_features(df, station_id, precip_fc_col, is_forecast=True)
        
        # Evapotranspiração
        df = self.add_evapotranspiration_features(df, station_id, et_col)
        
        # API
        df = self.add_api_features(df, station_id, precip_col, precip_fc_col)
        
        # Features avançadas
        df = self.add_advanced_features(df, station_id, flow_col, precip_col, 
                                       precip_fc_col, train_date_cutoff)
        
        return df
    
    def process_multiple_stations(self,
                                 data_dict: Dict[int, pd.DataFrame],
                                 train_date_cutoff: Optional[str] = None) -> pd.DataFrame:
        """
        Processa múltiplas estações e combina em um único DataFrame.
        
        Args:
            data_dict: Dicionário {station_id: DataFrame}
            train_date_cutoff: Data de corte para estatísticas
            
        Returns:
            DataFrame combinado com todas as features
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
        
        # Adicionar features sazonais (apenas uma vez)
        combined_df = self.add_seasonal_features(combined_df)
        
        return combined_df


def load_station_data(complete_series_dir: pathlib.Path, 
                     station_ids: List[int]) -> Dict[int, pd.DataFrame]:
    """
    Carrega dados de múltiplas estações da pasta complete_series.
    
    Args:
        complete_series_dir: Diretório com séries completas
        station_ids: Lista de IDs das estações
        
    Returns:
        Dicionário {station_id: DataFrame}
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


def process_features(input_dir: pathlib.Path,
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
                    output_filename: str = "features_combined.csv") -> pd.DataFrame:
    """
    Função principal para processamento de features.
    
    Args:
        input_dir: Diretório com séries completas (complete_series)
        output_dir: Diretório para salvar resultados (processed)
        station_ids: Lista de IDs das estações para processar
        api_k_list: Lista de valores k para API
        precipitation_ma_windows: Janelas para médias móveis de precipitação
        precipitation_cumulative_windows: Janelas para acumulados de precipitação
        forecast_ma_windows: Janelas para médias móveis de forecast
        forecast_cumulative_windows: Janelas para acumulados de forecast
        evapotranspiration_ma_windows: Janelas para médias móveis de evapotranspiração
        anomaly_ma_windows: Janelas para médias móveis de anomalias
        train_date_cutoff: Data de corte para estatísticas
        output_filename: Nome do arquivo de saída
        
    Returns:
        DataFrame com features processadas
    """
    print("Iniciando processamento de features...")
    
    # Verificar diretórios
    if not input_dir.exists():
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Carregar dados
    print(f"\nCarregando {len(station_ids)} estações...")
    data_dict = load_station_data(input_dir, station_ids)
    
    if not data_dict:
        raise ValueError("Nenhum dado foi carregado. Verifique os IDs das estações.")
    
    print(f"✓ {len(data_dict)} estações carregadas")
    
    # Processar features
    print("\nCriando features...")
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
        data_dict, 
        train_date_cutoff=train_date_cutoff
    )
    
    # Salvar resultados
    output_path = output_dir / output_filename
    combined_df.to_csv(output_path)
    
    print(f"\n✓ Features criadas com sucesso!")
    print(f"  Estações processadas: {len(data_dict)}")
    print(f"  Período: {combined_df.index.min().date()} a {combined_df.index.max().date()}")
    print(f"  Total de dias: {len(combined_df)}")
    print(f"  Total de features: {len(combined_df.columns)}")
    print(f"  Arquivo salvo: {output_path}")
    
    return combined_df