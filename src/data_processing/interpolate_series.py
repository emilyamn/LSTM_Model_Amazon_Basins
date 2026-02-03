"""
Módulo de interpolação IDW para dados hidrológicos.
"""

import os
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np


def load_station_data(file_path: str) -> pd.DataFrame:
    """
    Carrega dados de uma estação hidrológica.
    
    Args:
        file_path: Caminho do arquivo CSV
        
    Returns:
        DataFrame com dados carregados
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def calculate_idw_weights(distances: List[float], power: float = 1.0) -> np.ndarray:
    """
    Calcula pesos para interpolação IDW.
    
    Args:
        distances: Lista de distâncias em km
        power: Potência da distância (default: 1.0)
        
    Returns:
        Array de pesos normalizados
    """
    distances = np.array(distances, dtype=float)
    distances = np.where(distances <= 0, 1e-6, distances)  # Evitar divisão por zero
    weights = 1.0 / (distances ** power)
    return weights / weights.sum()


def interpolate_variable_idw(
    target_dates: pd.Series,
    neighbor_data: List[np.ndarray],
    distances: List[float],
    power: float = 1.0
) -> np.ndarray:
    """
    Aplica interpolação IDW para uma variável.
    
    Args:
        target_dates: Série com datas alvo
        neighbor_data: Lista de arrays com dados dos vizinhos
        distances: Distâncias dos vizinhos em km
        power: Potência da distância
        
    Returns:
        Array com valores interpolados
    """
    n_days = len(target_dates)
    interpolated = np.full(n_days, np.nan)
    
    if not neighbor_data:
        return interpolated
    
    # Calcular pesos
    weights = calculate_idw_weights(distances, power)
    
    # Criar matriz de dados
    data_matrix = np.column_stack(neighbor_data) if len(neighbor_data) > 1 else np.array(neighbor_data).reshape(-1, 1)
    
    # Aplicar IDW para cada dia
    for i in range(n_days):
        row = data_matrix[i, :]
        valid_mask = ~np.isnan(row)
        
        if not valid_mask.any():
            continue
        
        valid_weights = weights[valid_mask]
        valid_weights = valid_weights / valid_weights.sum()  # Renormalizar
        
        interpolated[i] = np.sum(row[valid_mask] * valid_weights)
    
    return interpolated


def interpolate_and_overwrite(
    target_station_id: int,
    variables: List[str],
    neighbor_station_ids: List[int],
    neighbor_distances_km: List[float],
    target_data_dir: str,
    neighbor_data_dir: str,
    power: float = 1.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Interpola variáveis para uma estação alvo e sobrescreve o arquivo original.
    
    Args:
        target_station_id: ID da estação alvo
        variables: Lista de variáveis para interpolar
        neighbor_station_ids: IDs das estações vizinhas
        neighbor_distances_km: Distâncias das estações vizinhas em km
        target_data_dir: Diretório com dados das estações alvo (complete_series)
        neighbor_data_dir: Diretório com dados das estações vizinhas (auxiliary_complete_series)
        power: Potência da distância para IDW
        verbose: Se True, imprime informações detalhadas
        
    Returns:
        DataFrame com dados interpolados
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"INTERPOLAÇÃO - Estação {target_station_id}")
        print(f"{'='*60}")
        print(f"Variáveis: {variables}")
        print(f"Vizinhos: {neighbor_station_ids}")
        print(f"Distâncias: {neighbor_distances_km} km")
        print(f"Power IDW: {power}")
    
    # 1. Carregar dados da estação alvo (da pasta complete_series)
    target_file = os.path.join(target_data_dir, f"{target_station_id}_complete_date.csv")
    if verbose:
        print(f"\nCarregando estação alvo: {os.path.basename(target_file)}")
    
    target_df = load_station_data(target_file)
    target_dates = target_df['date']
    
    if verbose:
        print(f"  Período: {target_dates.min().date()} a {target_dates.max().date()}")
        print(f"  Total dias: {len(target_df)}")
    
    # 2. Preparar dados dos vizinhos (da pasta auxiliary_complete_series)
    neighbor_data_dict = {}
    
    for nid, dist in zip(neighbor_station_ids, neighbor_distances_km):
        neighbor_file = os.path.join(neighbor_data_dir, f"{nid}_complete_date.csv")
        
        if os.path.exists(neighbor_file):
            try:
                df_neighbor = load_station_data(neighbor_file)
                neighbor_data_dict[nid] = df_neighbor
                if verbose:
                    print(f"  ✓ Vizinho {nid} carregado ({dist} km)")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Erro ao carregar vizinho {nid}: {e}")
        else:
            if verbose:
                print(f"  ✗ Arquivo do vizinho {nid} não encontrado")
    
    if not neighbor_data_dict:
        if verbose:
            print("AVISO: Nenhum vizinho carregado. Retornando dados originais.")
        return target_df
    
    # 3. Interpolar cada variável
    for variable in variables:
        if verbose:
            print(f"\n--- Interpolando '{variable}' ---")
        
        # Coletar dados dos vizinhos disponíveis
        neighbor_arrays = []
        available_neighbors = []
        available_distances = []
        
        for nid, df_neighbor in neighbor_data_dict.items():
            if variable in df_neighbor.columns:
                # Encontrar índice para obter a distância correta
                idx = neighbor_station_ids.index(nid)
                distance = neighbor_distances_km[idx]
                
                # Alinhar datas com o alvo
                aligned = pd.merge(
                    target_df[['date']],
                    df_neighbor[['date', variable]],
                    on='date',
                    how='left'
                )
                
                neighbor_arrays.append(aligned[variable].values)
                available_neighbors.append(nid)
                available_distances.append(distance)
                
                if verbose:
                    n_valid = np.sum(~np.isnan(aligned[variable].values))
                    print(f"  ✓ Vizinho {nid}: {n_valid}/{len(target_df)} valores")
            else:
                if verbose:
                    print(f"  ✗ Variável '{variable}' não encontrada no vizinho {nid}")
        
        if not available_neighbors:
            if verbose:
                print(f"  ⚠ Nenhum vizinho tem a variável '{variable}'")
            continue
        
        if verbose:
            print(f"  Total vizinhos disponíveis: {len(available_neighbors)}")
        
        # Aplicar IDW
        interpolated_values = interpolate_variable_idw(
            target_dates=target_dates,
            neighbor_data=neighbor_arrays,
            distances=available_distances,
            power=power
        )
        
        # Atualizar DataFrame alvo
        n_interpolated = np.sum(~np.isnan(interpolated_values))
        
        if variable in target_df.columns:
            # Preencher valores NaN com os interpolados
            existing_nan = target_df[variable].isna().sum()
            for i in range(len(target_df)):
                if pd.isna(target_df.at[i, variable]) and not pd.isna(interpolated_values[i]):
                    target_df.at[i, variable] = interpolated_values[i]
            
            if verbose:
                filled = existing_nan - target_df[variable].isna().sum()
                print(f"  Preenchidos: {filled} valores faltantes")
        else:
            # Adicionar nova coluna
            target_df[variable] = interpolated_values
            if verbose:
                print(f"  Adicionada nova coluna com {n_interpolated} valores")
        
        if verbose:
            n_final = target_df[variable].notna().sum()
            percentage = (n_final / len(target_df)) * 100
            print(f"  Resultado final: {n_final}/{len(target_df)} valores ({percentage:.1f}%)")
    
    # 4. SOBRESCREVER o arquivo original (mantendo o mesmo nome e local)
    target_df.to_csv(target_file, index=False)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ARQUIVO SOBRESCRITO: {target_file}")
        print(f"Colunas finais: {target_df.columns.tolist()}")
        print(f"{'='*60}")
    
    return target_df


def batch_interpolate_and_overwrite(
    station_configs: List[Dict],
    target_data_dir: str,
    neighbor_data_dir: str,
    verbose: bool = True
) -> Dict[int, pd.DataFrame]:
    """
    Executa interpolação em lote para múltiplas estações e sobrescreve arquivos.
    
    Args:
        station_configs: Lista de dicionários com configurações das estações
        target_data_dir: Diretório com dados das estações alvo (complete_series)
        neighbor_data_dir: Diretório com dados das estações vizinhas (auxiliary_complete_series)
        verbose: Se True, imprime informações detalhadas
        
    Returns:
        Dicionário com DataFrames processados
    """
    results = {}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"INTERPOLAÇÃO EM LOTE (COM SOBRESCRITA)")
        print(f"Total de estações: {len(station_configs)}")
        print(f"{'='*60}")
    
    for i, config in enumerate(station_configs, 1):
        if verbose:
            print(f"\n[{i}/{len(station_configs)}] Processando estação {config['target_id']}")
        
        try:
            result = interpolate_and_overwrite(
                target_station_id=config['target_id'],
                variables=config['variables'],
                neighbor_station_ids=config['neighbor_ids'],
                neighbor_distances_km=config['distances_km'],
                target_data_dir=target_data_dir,
                neighbor_data_dir=neighbor_data_dir,
                power=config.get('power', 1.0),
                verbose=verbose
            )
            
            results[config['target_id']] = result
            
        except Exception as e:
            if verbose:
                print(f"  ✗ Erro ao processar estação {config['target_id']}: {e}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"INTERPOLAÇÃO EM LOTE CONCLUÍDA")
        print(f"Estações processadas: {len(results)}/{len(station_configs)}")
        print(f"Arquivos atualizados na pasta: {target_data_dir}")
        print(f"{'='*60}")
    
    return results