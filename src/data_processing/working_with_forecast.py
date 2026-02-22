"""
Módulo para criar e manipular dados de forecast a partir de séries observadas.
Prepara dados climáticos para teste e produção do modelo.
"""

from typing import List, Dict
from datetime import timedelta
from pathlib import Path
import pandas as pd
import numpy as np


class ForecastGenerator:
    """
    Gera arquivos de forecast a partir de dados observados.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        extension_days: int = 18
    ):
        """
        Inicializa o gerador de forecast.

        Args:
            input_dir: Diretório com arquivos *_complete_date.csv
            output_dir: Diretório para salvar arquivos de forecast
            extension_days: Número de dias para estender além do último dado observado
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.extension_days = extension_days

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Diretório de entrada não encontrado: {self.input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_forecast_for_station(
        self,
        station_id: int,
        add_noise: bool = False,
        noise_std_precip: float = 0.1,
        noise_std_et: float = 0.05
    ) -> Dict[str, pd.DataFrame]:
        """
        Cria arquivos de forecast para uma estação específica.

        Args:
            station_id: ID da estação
            add_noise: Se True, adiciona ruído aos dados estendidos
            noise_std_precip: Desvio padrão do ruído para precipitação (fração do valor)
            noise_std_et: Desvio padrão do ruído para ET (fração do valor)

        Returns:
            Dicionário com DataFrames: {'precipitation': df, 'evapotranspiration': df}
        """
        # Carregar dados observados
        input_file = self.input_dir / f"{station_id}_complete_date.csv"

        if not input_file.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {input_file}")

        df = pd.read_csv(input_file, parse_dates=['date'])

        # Verificar colunas necessárias
        required_cols = ['date', 'precipitation_chirps', 'potential_evapotransp_gleam']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Colunas faltantes no arquivo: {missing_cols}")

        # Obter último dia com dados
        last_date = df['date'].max()

        # Criar período estendido
        extended_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=self.extension_days,
            freq='D'
        )

        # ==========================================
        # PRECIPITAÇÃO
        # ==========================================
        # Copiar dados observados
        df_precip = df[['date', 'precipitation_chirps']].copy()
        df_precip.columns = ['date', 'precipitation_forecast']
        df_precip['station_id'] = station_id

        # Estender com valores (última semana média ou persistência)
        last_week_precip = df['precipitation_chirps'].tail(7).mean()

        extended_precip = []
        for date in extended_dates:
            value = last_week_precip

            # Adicionar ruído se solicitado
            if add_noise:
                noise = np.random.normal(0, noise_std_precip * value)
                value = max(0, value + noise)  # Não permitir valores negativos

            extended_precip.append({
                'date': date,
                'station_id': station_id,
                'precipitation_forecast': value
            })

        df_extended_precip = pd.DataFrame(extended_precip)
        df_precip = pd.concat([df_precip, df_extended_precip], ignore_index=True)

        # ==========================================
        # EVAPOTRANSPIRAÇÃO
        # ==========================================
        # Copiar dados observados
        df_et = df[['date', 'potential_evapotransp_gleam']].copy()
        df_et.columns = ['date', 'et_forecast']
        df_et['station_id'] = station_id

        # Estender com valores (última semana média ou persistência)
        last_week_et = df['potential_evapotransp_gleam'].tail(7).mean()

        extended_et = []
        for date in extended_dates:
            value = last_week_et

            # Adicionar ruído se solicitado
            if add_noise:
                noise = np.random.normal(0, noise_std_et * value)
                value = max(0, value + noise)  # Não permitir valores negativos

            extended_et.append({
                'date': date,
                'station_id': station_id,
                'et_forecast': value
            })

        df_extended_et = pd.DataFrame(extended_et)
        df_et = pd.concat([df_et, df_extended_et], ignore_index=True)

        # ==========================================
        # SALVAR ARQUIVOS
        # ==========================================
        precip_file = self.output_dir / f"{station_id}_precipitation_forecast.csv"
        et_file = self.output_dir / f"{station_id}_evapotranspiration_forecast.csv"

        df_precip.to_csv(precip_file, index=False)
        df_et.to_csv(et_file, index=False)

        return {
            'precipitation': df_precip,
            'evapotranspiration': df_et,
            'precip_file': precip_file,
            'et_file': et_file
        }

    def create_forecast_for_multiple_stations(
        self,
        station_ids: List[int],
        add_noise: bool = False,
        noise_std_precip: float = 0.1,
        noise_std_et: float = 0.05
    ) -> Dict[int, Dict[str, pd.DataFrame]]:
        """
        Cria arquivos de forecast para múltiplas estações.

        Args:
            station_ids: Lista de IDs das estações
            add_noise: Se True, adiciona ruído aos dados estendidos
            noise_std_precip: Desvio padrão do ruído para precipitação
            noise_std_et: Desvio padrão do ruído para ET

        Returns:
            Dicionário {station_id: {'precipitation': df, 'evapotranspiration': df}}
        """
        results = {}

        print("="*60)
        print("GERAÇÃO DE ARQUIVOS DE FORECAST")
        print("="*60)
        print(f"Estações: {len(station_ids)}")
        print(f"Extensão: +{self.extension_days} dias")
        print(f"Adicionar ruído: {'Sim' if add_noise else 'Não'}")
        print()

        for station_id in station_ids:
            print(f"📊 Processando estação {station_id}...")

            try:
                result = self.create_forecast_for_station(
                    station_id=station_id,
                    add_noise=add_noise,
                    noise_std_precip=noise_std_precip,
                    noise_std_et=noise_std_et
                )

                results[station_id] = result

                # Estatísticas
                df_precip = result['precipitation']
                df_et = result['evapotranspiration']

                print("✅ Precipitação:")
                print(f"Período: {df_precip['date'].min().date()} a {df_precip['date'].max().date()}")
                print(f"Registros: {len(df_precip)}")
                print(f"Arquivo: {result['precip_file'].name}")

                print("   ✅ Evapotranspiração:")
                print(f"Período: {df_et['date'].min().date()} a {df_et['date'].max().date()}")
                print(f"Registros: {len(df_et)}")
                print(f"Arquivo: {result['et_file'].name}")
                print()

            except Exception as e:
                print(f"   ❌ Erro: {str(e)}")
                print()
                continue

        print("="*60)
        print("✅ GERAÇÃO CONCLUÍDA")
        print(f"   Estações processadas: {len(results)}/{len(station_ids)}")
        print(f"   Arquivos salvos em: {self.output_dir}")
        print("="*60)

        return results

    @staticmethod
    def load_forecast_data(
        forecast_dir: Path,
        station_ids: List[int]
    ) -> Dict[str, pd.DataFrame]:
        """
        Carrega dados de forecast para múltiplas estações e consolida.

        Args:
            forecast_dir: Diretório com arquivos de forecast
            station_ids: Lista de IDs das estações

        Returns:
            Dicionário com DataFrames consolidados:
            {'precipitation': df_all, 'evapotranspiration': df_all}
        """
        all_precip = []
        all_et = []

        for station_id in station_ids:
            precip_file = forecast_dir / f"{station_id}_precipitation_forecast.csv"
            et_file = forecast_dir / f"{station_id}_evapotranspiration_forecast.csv"

            # Carregar precipitação
            if precip_file.exists():
                df_precip = pd.read_csv(precip_file, parse_dates=['date'])
                all_precip.append(df_precip)
            else:
                print(f"⚠️  Arquivo não encontrado: {precip_file}")

            # Carregar ET
            if et_file.exists():
                df_et = pd.read_csv(et_file, parse_dates=['date'])
                all_et.append(df_et)
            else:
                print(f"⚠️  Arquivo não encontrado: {et_file}")

        # Consolidar
        result = {}

        if all_precip:
            df_precip_all = pd.concat(all_precip, ignore_index=True)
            df_precip_all = df_precip_all.sort_values(['date', 'station_id'])
            result['precipitation'] = df_precip_all

        if all_et:
            df_et_all = pd.concat(all_et, ignore_index=True)
            df_et_all = df_et_all.sort_values(['date', 'station_id'])
            result['evapotranspiration'] = df_et_all

        return result


def generate_forecast_files(
    input_dir: Path,
    output_dir: Path,
    station_ids: List[int],
    extension_days: int = 18,
    add_noise: bool = False,
    noise_std_precip: float = 0.1,
    noise_std_et: float = 0.05
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """
    Função de conveniência para gerar arquivos de forecast.

    Args:
        input_dir: Diretório com arquivos *_complete_date.csv
        output_dir: Diretório para salvar arquivos de forecast
        station_ids: Lista de IDs das estações
        extension_days: Número de dias para estender além do último dado
        add_noise: Se True, adiciona ruído aos dados estendidos
        noise_std_precip: Desvio padrão do ruído para precipitação (fração)
        noise_std_et: Desvio padrão do ruído para ET (fração)

    Returns:
        Dicionário com resultados por estação
    """
    generator = ForecastGenerator(
        input_dir=input_dir,
        output_dir=output_dir,
        extension_days=extension_days
    )

    return generator.create_forecast_for_multiple_stations(
        station_ids=station_ids,
        add_noise=add_noise,
        noise_std_precip=noise_std_precip,
        noise_std_et=noise_std_et
    )
