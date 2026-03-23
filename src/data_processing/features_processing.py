"""
Módulo para feature engineering de dados hidrológicos.
"""

from typing import Dict, List, Optional, Literal
import pathlib
import sys
import pandas as pd
import numpy as np

# Corrigir a importação do config_loader
current_dir = pathlib.Path(__file__).parent
project_root = current_dir.parent.parent

# Adicionar o src ao sys.path
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Tipo para forçantes
ForcingType = Literal["P", "P_ET"]

class HydroFeatureEngineer:
    """
    Classe para criação de features hidrológicas avançadas.

    Supports two forcing configurations:
    - "P": Only precipitation (default) - for operational use without ET
    - "P_ET": Precipitation + Evapotranspiration - for training with ET data
    """

    def __init__(self,
                 api_k_list: List[float] = None,
                 precipitation_ma_windows: List[int] = None,
                 precipitation_cumulative_windows: List[int] = None,
                 forecast_ma_windows: List[int] = None,
                 forecast_cumulative_windows: List[int] = None,
                 evapotranspiration_ma_windows: List[int] = None,
                 anomaly_ma_windows: List[int] = None,
                 forcings: ForcingType = "P"):
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
            forcings: Forçantes do modelo
                - "P": apenas precipitação (padrão, para operação sem ET)
                - "P_ET": precipitação + evapotranspiração (para treino com ET)
        """
        # Valores padrão
        self.api_k_list = api_k_list or [0.70, 0.80, 0.85, 0.90, 0.92, 0.95]
        self.precipitation_ma_windows = precipitation_ma_windows or [3, 7, 15]
        self.precipitation_cumulative_windows = precipitation_cumulative_windows or [3, 5, 7, 10]
        self.forecast_ma_windows = forecast_ma_windows or [3, 7, 15]
        self.forecast_cumulative_windows = forecast_cumulative_windows or [3, 5, 7, 10]
        self.evapotranspiration_ma_windows = evapotranspiration_ma_windows or [7, 14, 30]
        self.anomaly_ma_windows = anomaly_ma_windows or [3, 7]

        # Forçantes - define quais variáveis usar
        self.forcings = forcings if forcings else "P"

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
                        precip_obs_col: Optional[str],
                        precip_forecast_col: Optional[str]) -> pd.DataFrame:
        """
        Adiciona features de API para dados observados e forecast.
        """
        for k in self.api_k_list:
            tag = f"k{int(round(k*100)):02d}"

            # API observada - apenas se a coluna existir
            if precip_obs_col and precip_obs_col in df.columns:
                df[f'api_obs_{tag}_{station}'] = self.compute_api(df[precip_obs_col], k)

            # API forecast - apenas se a coluna existir
            if precip_forecast_col and precip_forecast_col in df.columns:
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
                             precip_obs_col: Optional[str],
                             precip_forecast_col: Optional[str],
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

        # Derivada do fluxo - sempre
        df[f'dQ_dt_{station}'] = df[flow_col].diff().fillna(0.0)

        # Derivadas de precipitação - apenas se existirem
        if precip_obs_col and precip_obs_col in df.columns:
            df[f'dP_obs_dt_{station}'] = df[precip_obs_col].diff().fillna(0.0)

        if precip_forecast_col and precip_forecast_col in df.columns:
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

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Método de conveniência que chama process_station para um único DataFrame.
        """
        return df

    def process_station(
        self,
        df: pd.DataFrame,
        station_id: int,
        train_date_cutoff: Optional[str] = None,
        column_names: Optional[Dict[str, str]] = None,
        forcings: Optional[ForcingType] = None
    ) -> pd.DataFrame:
        """
        ⚙️ Função auxiliar - processa UMA estação.
        Chamada internamente por process_stations.

        Args:
            df: DataFrame com os dados
            station_id: ID da estação
            train_date_cutoff: Data de corte para treino
            column_names: Mapeamento de colunas
            forcings: Forçantes ("P" ou "P_ET"). Se None, usa o padrão da classe
        """
        # Usar forcings da classe se não especificado
        use_forcings = forcings if forcings is not None else self.forcings

        # Definir nomes padrão
        default_column_names = {
            'flow': 'streamflow_m3s',
            'precip_obs': 'precipitation_chirps',
            'et_obs': 'potential_evapotransp_gleam',
            'precip_forecast': 'precipitation_forecast',
            'et_forecast': 'et_forecast'
        }

        col_map = column_names if column_names is not None else default_column_names

        # Verificar APENAS colunas obrigatórias (flow)
        if 'flow' not in col_map:
            raise ValueError(f"Estação {station_id}: Coluna 'flow' é obrigatória no column_names")

        flow_col = col_map['flow']
        if flow_col not in df.columns:
            raise ValueError(
                f"❌ Coluna de fluxo faltante para estação {station_id}: {flow_col}"
                f"   Disponíveis: {list(df.columns)}"
            )

        # Preparar DataFrame
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # Renomear APENAS as colunas que existem E que são permitidas pelo forcings
        rename_map = {}
        rename_map[col_map['flow']] = f'Q_{station_id}'

        # Precip observada (sempre disponível se existir)
        if 'precip_obs' in col_map and col_map['precip_obs'] in df.columns:
            rename_map[col_map['precip_obs']] = f'precipitation_obs_{station_id}'

        # Precip forecast (sempre disponível se existir)
        if 'precip_forecast' in col_map and col_map['precip_forecast'] in df.columns:
            rename_map[col_map['precip_forecast']] = f'precipitation_forecast_{station_id}'

        # ET: apenas se forcings="P_ET" e a coluna existir
        if use_forcings == "P_ET":
            if 'et_obs' in col_map and col_map['et_obs'] in df.columns:
                rename_map[col_map['et_obs']] = f'et_obs_{station_id}'
            if 'et_forecast' in col_map and col_map['et_forecast'] in df.columns:
                rename_map[col_map['et_forecast']] = f'et_forecast_{station_id}'

        df = df.rename(columns=rename_map)

        # Definir nomes internos (apenas se existirem)
        flow_col_renamed = f'Q_{station_id}'
        precip_obs_col = f'precipitation_obs_{station_id}' if f'precipitation_obs_{station_id}' in df.columns else None
        precip_fc_col = f'precipitation_forecast_{station_id}' if f'precipitation_forecast_{station_id}' in df.columns else None

        # ET: apenas se forcings="P_ET"
        et_obs_col = None
        et_fc_col = None
        if use_forcings == "P_ET":
            et_obs_col = f'et_obs_{station_id}' if f'et_obs_{station_id}' in df.columns else None
            et_fc_col = f'et_forecast_{station_id}' if f'et_forecast_{station_id}' in df.columns else None

        # Features - apenas para colunas que existem
        if precip_obs_col:
            df = self.add_precipitation_features(df, station_id, precip_obs_col, is_forecast=False)
        if precip_fc_col:
            df = self.add_precipitation_features(df, station_id, precip_fc_col, is_forecast=True)
        if et_obs_col:
            df = self.add_evapotranspiration_features(df, station_id, et_obs_col, is_forecast=False)
        if et_fc_col:
            df = self.add_evapotranspiration_features(df, station_id, et_fc_col, is_forecast=True)

        # API - apenas se tiver precip
        if precip_obs_col or precip_fc_col:
            df = self.add_api_features(df, station_id, precip_obs_col, precip_fc_col)

        # Advanced features - requer flow
        df = self.add_advanced_features(df, station_id, flow_col_renamed, precip_obs_col, precip_fc_col, train_date_cutoff)

        return df

    def process_multiple_stations(
        self,
        data_dict: Dict[int, pd.DataFrame],
        train_date_cutoff: Optional[str] = None,
        column_names: Optional[Dict[str, str]] = None,
        forcings: Optional[ForcingType] = None
    ) -> pd.DataFrame:
        """
        MÉTODO PRINCIPAL - Processa uma ou múltiplas estações.

        Args:
            data_dict: {station_id: DataFrame}s
            train_date_cutoff: Data de corte (opcional)
            column_names: Mapeamento de colunas (opcional)
            forcings: Forçantes ("P" ou "P_ET"). Se None, usa o padrão da classe

        Returns:
            DataFrame combinado com features
        """
        # Usar forcings da classe se não especificado
        use_forcings = forcings if forcings is not None else self.forcings

        if not data_dict:
            raise ValueError("data_dict vazio")

        processed_dfs = []

        print(f"{'='*60}")
        print(f"PROCESSANDO {len(data_dict)} ESTAÇÃO(ÕES)")
        print(f"   Forçantes: {use_forcings}")
        print(f"{'='*60}")

        for station_id, df in data_dict.items():
            try:
                print(f"📊 Estação {station_id}...")

                # Chama auxiliar COM column_names E forcings
                df_processed = self.process_station(
                    df=df,
                    station_id=station_id,
                    train_date_cutoff=train_date_cutoff,
                    column_names=column_names,
                    forcings=use_forcings
                )

                processed_dfs.append(df_processed)
                print(f"✅ OK - {len(df_processed.columns)} features")

            except Exception as e:
                print(f"❌ Erro: {e}")
                continue

        if not processed_dfs:
            raise ValueError("Nenhuma estação processada")

        # Combinar
        combined_df = processed_dfs[0]
        for df in processed_dfs[1:]:
            combined_df = combined_df.merge(df, left_index=True, right_index=True, how='outer')

        combined_df = combined_df.ffill().bfill()
        combined_df = self.add_seasonal_features(combined_df)

        print(f"✅ CONCLUÍDO - {len(combined_df.columns)} colunas")
        print(f"{'='*60}")

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


def load_forecast_data(
    forecast_dir: pathlib.Path,
    station_ids: List[int],
    forcings: ForcingType = "P"  # ← ADICIONAR PARÂMETRO
) -> Dict[int, pd.DataFrame]:
    """
    Carrega dados de forecast para múltiplas estações.
    
    Args:
        forecast_dir: Diretório com arquivos de forecast
        station_ids: Lista de IDs das estações
        forcings: "P" = só precipitação, "P_ET" = precipitação + ET
    """
    forecast_dict = {}

    for station_id in station_ids:
        precip_file = forecast_dir / f"{station_id}_precipitation_forecast.csv"
        et_file = forecast_dir / f"{station_id}_evapotranspiration_forecast.csv"

        # ✅ VERIFICAR se precisa carregar ET
        need_et = (forcings == "P_ET")
        
        # Verificar se arquivos existem
        precip_exists = precip_file.exists()
        et_exists = et_file.exists() if need_et else False
        
        if not precip_exists:
            print(f"✗ Arquivo de precipitação não encontrado para estação {station_id}")
            continue
        
        if need_et and not et_exists:
            print(f"⚠️  Arquivo de ET não encontrado para estação {station_id} (forcings=P_ET)")
            continue

        try:
            # Carregar precipitação (SEMPRE)
            df_precip = pd.read_csv(precip_file, parse_dates=['date'])
            df_precip['date'] = pd.to_datetime(df_precip['date'])
            df_precip = df_precip[['date', 'precipitation_forecast']]

            # ✅ Carregar ET APENAS se forcings="P_ET"
            if need_et:
                df_et = pd.read_csv(et_file, parse_dates=['date'])
                df_et['date'] = pd.to_datetime(df_et['date'])
                df_et = df_et[['date', 'et_forecast']]
                
                df_forecast = df_precip.merge(df_et, on='date', how='outer')
            else:
                # Apenas precipitação
                df_forecast = df_precip

            forecast_dict[station_id] = df_forecast

        except Exception as e:
            print(f"✗ Erro ao carregar forecast da estação {station_id}: {e}")

    return forecast_dict

def merge_observed_and_forecast(
    observed_dict: Dict[int, pd.DataFrame],
    forecast_dict: Dict[int, pd.DataFrame],
    forcings: ForcingType = "P"
) -> Dict[int, pd.DataFrame]:
    """
    Faz merge dos dados observados com forecast.
    """
    merged_dict = {}
    need_et = (forcings == "P_ET")

    for station_id in observed_dict.keys():
        if station_id not in forecast_dict:
            print(f"⚠️  Sem dados de forecast para estação {station_id}, usando observados")
            df_merged = observed_dict[station_id].copy()
            df_merged['precipitation_forecast'] = df_merged['precipitation_chirps']
            
            if need_et and 'potential_evapotransp_gleam' in df_merged.columns:
                df_merged['et_forecast'] = df_merged['potential_evapotransp_gleam']
            else:
                et_cols = [c for c in df_merged.columns if 'evapotransp' in c.lower() or 'et_' in c.lower()]
                if et_cols:
                    df_merged = df_merged.drop(columns=et_cols)
            
            merged_dict[station_id] = df_merged
            continue

        df_obs = observed_dict[station_id].copy()
        df_fc = forecast_dict[station_id].copy()

        df_obs['date'] = pd.to_datetime(df_obs['date'])
        df_fc['date'] = pd.to_datetime(df_fc['date'])

        df_merged = df_obs.merge(df_fc, on='date', how='left')

        df_merged['precipitation_forecast'] = df_merged['precipitation_forecast'].fillna(
            df_merged['precipitation_chirps']
        )
        
        if need_et:
            if 'et_forecast' in df_merged.columns and 'potential_evapotransp_gleam' in df_merged.columns:
                df_merged['et_forecast'] = df_merged['et_forecast'].fillna(
                    df_merged['potential_evapotransp_gleam']
                )
        else:
            # REMOVER colunas de ET
            et_cols_to_drop = [
                col for col in df_merged.columns 
                if any(keyword in col.lower() for keyword in ['evapotransp', 'et_forecast', 'et_obs', '_et_', 'gleam'])
            ]
            if et_cols_to_drop:
                df_merged = df_merged.drop(columns=et_cols_to_drop)
                print(f"   🗑️  Removidas {len(et_cols_to_drop)} colunas de ET (forcings=P)")

        # ✅✅✅ REMOVER COLUNAS DESNECESSÁRIAS DO MERGE (station_id, missing_data, etc.)
        cols_to_drop = [
            col for col in df_merged.columns 
            if col.endswith('_x') or col.endswith('_y') or 
               col in ['station_id', 'missing_data', 'Unnamed: 0']
        ]
        if cols_to_drop:
            df_merged = df_merged.drop(columns=cols_to_drop)
            print(f"   🗑️  Removidas colunas desnecessárias: {cols_to_drop}")

        merged_dict[station_id] = df_merged

    return merged_dict

def process_features(
    input_dir: pathlib.Path,
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
    forcings: ForcingType = "P"
) -> pd.DataFrame:
    """
    Função principal para processamento de features.

    Args:
        forcings: "P" = apenas precipitação (padrão, para operação)
                   "P_ET" = precipitação + ET (para treino com ET)
    """
    print("="*60)
    print("PROCESSAMENTO DE FEATURES")
    print(f"   Forçantes: {forcings}")
    print("="*60)

    if not input_dir.exists():
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_dir}")
    if not forecast_dir.exists():
        raise FileNotFoundError(f"Diretório de forecast não encontrado: {forecast_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 Carregando dados observados de {len(station_ids)} estações...")
    observed_dict = load_station_data(input_dir, station_ids)
    print(f"✓ {len(observed_dict)} estações carregadas")

    print("🔮 Carregando dados de forecast...")
    # ✅ PASSAR forcings
    forecast_dict = load_forecast_data(forecast_dir, station_ids, forcings=forcings)
    print(f"✓ {len(forecast_dict)} estações com forecast carregadas")

    print("🔗 Fazendo merge observado + forecast...")
    # ✅ PASSAR forcings
    merged_dict = merge_observed_and_forecast(observed_dict, forecast_dict, forcings=forcings)
    print(f"✓ {len(merged_dict)} estações combinadas")

    if not merged_dict:
        raise ValueError("Nenhum dado foi combinado. Verifique os arquivos de forecast.")

    print("⚙️  Criando features...")
    engineer = HydroFeatureEngineer(
        api_k_list=api_k_list,
        precipitation_ma_windows=precipitation_ma_windows,
        precipitation_cumulative_windows=precipitation_cumulative_windows,
        forecast_ma_windows=forecast_ma_windows,
        forecast_cumulative_windows=forecast_cumulative_windows,
        evapotranspiration_ma_windows=evapotranspiration_ma_windows,
        anomaly_ma_windows=anomaly_ma_windows,
        forcings=forcings  # ← Já estava correto
    )

    combined_df = engineer.process_multiple_stations(
        merged_dict,
        train_date_cutoff=train_date_cutoff,
        forcings=forcings  # ← Já estava correto
    )

    output_path = output_dir / output_filename
    combined_df.to_csv(output_path)

    print("" + "="*60)
    print("✅ FEATURES CRIADAS COM SUCESSO")
    print("="*60)
    print(f"  Forçantes: {forcings}")
    print(f"  Estações processadas: {len(merged_dict)}")
    print(f"  Período: {combined_df.index.min().date()} a {combined_df.index.max().date()}")
    print(f"  Total de dias: {len(combined_df)}")
    print(f"  Total de features: {len(combined_df.columns)}")
    print(f"  Arquivo salvo: {output_path}")
    print("="*60)

    return combined_df
