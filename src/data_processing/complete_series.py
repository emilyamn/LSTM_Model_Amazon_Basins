"""
Módulo para completar séries temporais de estações hidrológicas.
"""

import os
import pandas as pd
from typing import List, Dict, Optional, Tuple


class DataPreprocessor:
    """Classe para pré-processamento de dados hidrológicos"""
    
    def __init__(self, csv_path: str):
        """
        Inicializa o preprocessador.
        
        Args:
            csv_path: Caminho para os arquivos CSV originais
        """
        self.csv_path = csv_path
        self.amazon_data = {}
        self.processed_data = {}
        
    def load_station(self, station_id: int) -> Optional[pd.DataFrame]:
        """
        Carrega dados de uma única estação.
        
        Args:
            station_id: ID da estação
            
        Returns:
            DataFrame com dados da estação ou None se não encontrado
        """
        arquivo = os.path.join(self.csv_path, f"{station_id}.csv")
        
        if os.path.exists(arquivo):
            return pd.read_csv(arquivo)
        else:
            print(f"Arquivo não encontrado: {arquivo}")
            return None
    
    def load_multiple_stations(self, station_ids: List[int]) -> Dict[int, pd.DataFrame]:
        """
        Carrega múltiplas estações.
        
        Args:
            station_ids: Lista de IDs das estações
            
        Returns:
            Dicionário com DataFrames carregados
        """
        self.amazon_data = {}
        for station_id in station_ids:
            df = self.load_station(station_id)
            if df is not None:
                self.amazon_data[station_id] = df
        return self.amazon_data
    
    def process_station(self, df_original: pd.DataFrame,
                       station_id: int,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Processa uma única estação, preenchendo datas faltantes.

        Args:
            df_original: DataFrame original da estação
            station_id: ID da estação
            start_date: Data inicial do período. Se None, usa a primeira data
                        da própria série (permite estações com tamanhos
                        diferentes).
            end_date:   Data final do período. Se None, usa a última data da
                        própria série.

        Returns:
            DataFrame processado com datas completas
        """
        df = df_original.copy()
        df['date'] = pd.to_datetime(df['date'])

        effective_start = pd.to_datetime(start_date) if start_date is not None else df['date'].min()
        effective_end   = pd.to_datetime(end_date)   if end_date   is not None else df['date'].max()

        # Criar intervalo completo de datas
        date_range = pd.DataFrame({
            "date": pd.date_range(effective_start, effective_end, freq="D")
        })
        
        # Fazer merge para completar datas
        df_full = pd.merge(date_range, df, on="date", how="left")
        
        # Adicionar identificador da estação
        df_full['station_id'] = station_id
        
        # Identificar onde não há correspondência
        df_full["missing_data"] = df_full.isna().any(axis=1)
        
        return df_full
    
    def process_and_save_stations(self,
                                 main_stations: List[int],
                                 auxiliary_stations: List[int],
                                 main_output_path: str,
                                 auxiliary_output_path: str,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> Tuple[Dict, Dict]:
        """
        Processa e salva estações em pastas separadas.

        Args:
            main_stations: Lista de IDs para complete_series
            auxiliary_stations: Lista de IDs para auxiliary_complete_series
            main_output_path: Caminho para salvar séries principais
            auxiliary_output_path: Caminho para salvar séries auxiliares
            start_date: Data inicial global. Se None, cada estação usa a sua
                        própria primeira data (tamanhos diferentes ok).
            end_date:   Data final global. Se None, cada estação usa a sua
                        própria última data.
            
        Returns:
            Tupla com (dicionário de séries principais, dicionário de séries auxiliares)
        """
        # Carregar todas as estações necessárias
        all_stations = list(set(main_stations + auxiliary_stations))
        self.load_multiple_stations(all_stations)
        
        # Processar séries principais
        main_data = {}
        for station_id in main_stations:
            if station_id in self.amazon_data:
                df_processed = self.process_station(
                    self.amazon_data[station_id], station_id, start_date, end_date
                )
                main_data[station_id] = df_processed
                
                # Salvar na pasta principal
                os.makedirs(main_output_path, exist_ok=True)
                output_file = os.path.join(main_output_path, f"{station_id}_complete_date.csv")
                df_processed.to_csv(output_file, index=False)
                print(f"✓ Estação {station_id} salva em: {output_file}")
        
        # Processar séries auxiliares
        auxiliary_data = {}
        for station_id in auxiliary_stations:
            if station_id in self.amazon_data:
                df_processed = self.process_station(
                    self.amazon_data[station_id], station_id, start_date, end_date
                )
                auxiliary_data[station_id] = df_processed
                
                # Salvar na pasta auxiliar
                os.makedirs(auxiliary_output_path, exist_ok=True)
                output_file = os.path.join(auxiliary_output_path, f"{station_id}_complete_date.csv")
                df_processed.to_csv(output_file, index=False)
                print(f"✓ Estação {station_id} (auxiliar) salva em: {output_file}")
        
        return main_data, auxiliary_data