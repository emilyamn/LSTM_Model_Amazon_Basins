"""
Módulo para gerenciamento de experimentos de forma organizada e reprodutível.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import subprocess
import json
import shutil
import re
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import torch

def convert_predictions_to_df(
    preds,
    reference_dates: List,
    stations: List[int],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Converte o array de previsões em um DataFrame (sem salvar).

    Args:
        preds: Array de previsões (batch, horizonte, n_estacoes)
        reference_dates: Lista de datas de referência
        stations: Lista de códigos das estações

    Returns:
        DataFrame no formato:
        date | Q_pred_{station1} | Q_pred_{station2} | ...
    """

    # Remove dimensão de batch
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    preds = preds.squeeze(0)  # (horizonte, n_estacoes)

    horizonte = preds.shape[0]

    # Data inicial (última observada)
    ref_date = pd.to_datetime(reference_dates[0])

    # Cria lista de datas futuras (D+1 até D+horizonte)
    datas_futuras = [ref_date + pd.Timedelta(days=i) for i in range(1, horizonte + 1)]

    # Cria DataFrame
    df = pd.DataFrame(preds, columns=[f"Q_pred_{s}" for s in stations])

    # Adiciona coluna de data
    df.insert(0, 'date', datas_futuras)

    # Salvar se caminho foi fornecido
    if save_path is not None:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Nome do arquivo = prediction_YYYY_MM_DD.csv (D+1)
        data_inicio = datas_futuras[0]
        nome_arquivo = f"prediction_{data_inicio.strftime('%Y_%m_%d')}.csv"

        file_path = save_dir / nome_arquivo
        df.to_csv(file_path, index=False, encoding='utf-8')

        print(f"💾 Previsões salvas: {file_path}")

    return df

def get_project_root() -> Path:
    """
    Encontra a raiz do projeto (onde está a pasta src/).

    Returns:
        Path da raiz do projeto
    """
    # Assumindo que este arquivo está em src/utils/
    current_file = Path(__file__).resolve()
    # src/utils/experiment_utils.py -> src/utils/ -> src/ -> projeto/
    project_root = current_file.parent.parent.parent
    return project_root


def get_experiments_base_dir() -> Path:
    """
    Retorna o diretório base para experimentos.

    Returns:
        Path de outputs/experiments/
    """
    return get_project_root() / "outputs" / "experiments"


def get_next_experiment_id(base_dir: Optional[Path] = None) -> int:
    """
    Encontra o próximo ID de experimento disponível.

    Args:
        base_dir: Diretório base onde estão os experimentos (opcional)

    Returns:
        Próximo ID disponível (ex: se existe exp_001, retorna 2)
    """
    if base_dir is None:
        base_dir = get_experiments_base_dir()

    if not base_dir.exists():
        return 1

    existing = [d.name for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")]

    if not existing:
        return 1

    # Extrair IDs existentes usando regex (mais robusto)
    ids = []
    for exp_name in existing:
        match = re.match(r"^exp_(\d+)", exp_name)
        if match:
            ids.append(int(match.group(1)))

    return max(ids) + 1 if ids else 1

def get_git_branch() -> str:
    """
    Obtém o nome da branch atual do Git.

    Returns:
        Nome da branch sanitizado (caracteres especiais removidos)
        Retorna "no-git" se não estiver em um repositório Git
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=get_project_root(),
            timeout=5,
            check=False  # ✅ ADICIONE ESTA LINHA
        )

        if result.returncode == 0:
            branch = result.stdout.strip()

            # Sanitizar o nome da branch
            # Substituir / por - (ex: feature/new-model -> feature-new-model)
            branch = branch.replace('/', '-').replace('\\', '-')

            # Remover outros caracteres problemáticos
            branch = re.sub(r'[^a-zA-Z0-9_-]', '', branch)

            # Limitar tamanho
            if len(branch) > 30:
                branch = branch[:30]

            return branch if branch else "no-git"
        else:
            return "no-git"
    except Exception as e:
        print(f"⚠️  Não foi possível detectar branch Git: {e}")
        return "no-git"


def find_experiment_by_name(
    experiment_name: str,
    branch_name: Optional[str] = None,
    base_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Encontra o experimento mais recente com um determinado nome (e opcionalmente branch).

    Args:
        experiment_name: Nome do experimento (sem o prefixo exp_XXX_branch_)
        branch_name: Nome da branch (opcional). Se None, usa a branch atual.
        base_dir: Diretório base onde estão os experimentos (opcional)

    Returns:
        Path do experimento mais recente ou None se não encontrado
    """
    if base_dir is None:
        base_dir = get_experiments_base_dir()

    if not base_dir.exists():
        return None

    if branch_name is None:
        branch_name = get_git_branch()

    # Usa regex para parsing correto do formato exp_XXX_branchName_experimentName
    pattern = re.compile(rf"^exp_(\d+)_{re.escape(branch_name)}_(.+)$")

    matching = []
    for exp_dir in base_dir.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
            continue

        match = pattern.match(exp_dir.name)
        if match:
            exp_id = int(match.group(1))
            exp_name = match.group(2)
            if exp_name == experiment_name:
                matching.append((exp_id, exp_dir))

    if not matching:
        # Se não encontrou com a branch especificada, avisar
        print(f"⚠️  Experimento '{experiment_name}' não encontrado na branch '{branch_name}'")
        print("    Procurando em outras branches...")

        # Tentar encontrar em qualquer branch
        for exp_dir in base_dir.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
                continue

            parts = exp_dir.name.split("_", 3)
            if len(parts) >= 4:
                try:
                    exp_id = int(parts[1])
                    exp_name = parts[3]

                    if exp_name == experiment_name:
                        matching.append((exp_id, exp_dir))
                except ValueError:
                    continue

        if not matching:
            return None

        # Avisar que encontrou em outra branch
        matching.sort(key=lambda x: x[0], reverse=True)
        found_exp = matching[0][1]
        found_branch = found_exp.name.split("_", 3)[2]
        print(f"    ✅ Encontrado em branch '{found_branch}': {found_exp.name}")
        return found_exp

    # Retornar o mais recente (maior ID)
    matching.sort(key=lambda x: x[0], reverse=True)
    return matching[0][1]


def get_experiment_path(
    experiment_name: str,
    base_dir: Optional[Path] = None
) -> Path:
    """
    Obtém o path completo de um experimento pelo nome.

    Args:
        experiment_name: Nome do experimento
        base_dir: Diretório base onde estão os experimentos (opcional)

    Returns:
        Path do experimento

    Raises:
        FileNotFoundError: Se experimento não existir
    """
    exp_path = find_experiment_by_name(experiment_name, base_dir=base_dir)

    if exp_path is None:
        raise FileNotFoundError(
            f"Experimento '{experiment_name}' não encontrado. "
            f"Certifique-se de criar o experimento primeiro usando create_experiment()."
        )

    return exp_path


def merge_configs(config_paths: Dict[str, Path]) -> Dict[str, Any]:
    """
    Faz merge de múltiplos arquivos YAML de configuração.

    Args:
        config_paths: Dicionário com paths dos arquivos YAML
                     Ex: {"data": Path, "model": Path, "training": Path}

    Returns:
        Dicionário com todas as configurações merged
    """
    merged = {}

    for key, path in config_paths.items():
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Adicionar sob a chave correspondente
        merged[key] = config

    return merged


def create_experiment(
    config_paths: Dict[str, Path],
    experiment_name: str,
    description: str = "",
    base_dir: Optional[Path] = None
) -> Path:
    """
    Cria uma nova pasta de experimento com estrutura organizada.

    Args:
        config_paths: Dicionário com paths dos arquivos YAML originais
                     Pode ser relativo à raiz do projeto ou absoluto
                     Ex: {"data": "config/data_config.yaml",
                          "model": "config/model_config.yaml",
                          "training": "config/training_config.yaml"}
        experiment_name: Nome do experimento (ex: "baseline_lstm")
        description: Descrição do experimento (opcional)
        base_dir: Diretório base para experimentos (opcional, padrão: outputs/experiments)

    Returns:
        Path do experimento criado

    Raises:
        FileNotFoundError: Se algum arquivo de config não existir
    """
    # Obter raiz do projeto
    project_root = get_project_root()

    # Obter nome da branch
    branch_name = get_git_branch()
    print(f"📌 Branch detectada: {branch_name}")

    # Converter paths para Path e resolver relativos à raiz do projeto
    resolved_paths = {}
    for key, path in config_paths.items():
        path_obj = Path(path)

        # Se for relativo, resolver em relação à raiz do projeto
        if not path_obj.is_absolute():
            path_obj = project_root / path_obj

        resolved_paths[key] = path_obj

    # Validar que todos os arquivos existem
    for key, path in resolved_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                f"Project root: {project_root}\n"
                f"Tried to find: {key} config at {path}"
            )

    # Usar diretório padrão se não especificado
    if base_dir is None:
        base_dir = get_experiments_base_dir()

    # Criar diretório base se não existir
    base_dir.mkdir(parents=True, exist_ok=True)

    # Gerar ID e nome do experimento (incluindo branch)
    exp_id = get_next_experiment_id(base_dir)
    exp_dir_name = f"exp_{exp_id:03d}_{branch_name}_{experiment_name}"
    exp_path = base_dir / exp_dir_name

    print(f"\n{'='*80}")
    print(f"CRIANDO NOVO EXPERIMENTO: {exp_dir_name}")
    print(f"{'='*80}\n")

    # Criar estrutura de diretórios
    subdirs = {
        "config": exp_path / "config",
        "model": exp_path / "model",
        "logs": exp_path / "logs",
        "test_raw": exp_path / "predictions_test" / "raw",
        "test_metrics": exp_path / "predictions_test" / "metrics",
        "test_plots": exp_path / "predictions_test" / "plots",
        "operational_raw": exp_path / "predictions_operational" / "raw",
        "operational_plots": exp_path / "predictions_operational" / "plots",
    }

    for name, path in subdirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Criado: {path.relative_to(base_dir)}")

    # Copiar arquivos de configuração originais
    print("\n📋 Copiando arquivos de configuração...")
    for key, src_path in resolved_paths.items():
        dest_path = subdirs["config"] / src_path.name
        shutil.copy2(src_path, dest_path)
        print(f"  ✓ {src_path.name}")

    # Fazer merge dos configs e salvar full_config.yaml
    print("\n🔀 Criando full_config.yaml (merged)...")
    merged_config = merge_configs(resolved_paths)

    full_config_path = subdirs["config"] / "full_config.yaml"
    with open(full_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
    print("  ✓ full_config.yaml")

    # Criar run_info.txt
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_info = f"""Experiment: {exp_dir_name}
ID: {exp_id:03d}
Branch: {branch_name}
Name: {experiment_name}
Created: {timestamp}

Description:
{description if description else "No description provided"}

Configuration Files:
{chr(10).join(f"- {p.name}" for p in resolved_paths.values())}
- full_config.yaml (merged)
"""

    run_info_path = subdirs["logs"] / "run_info.txt"
    with open(run_info_path, 'w', encoding='utf-8') as f:
        f.write(run_info)
    print("\n✓ Criado: logs/run_info.txt")

    # Salvar metadata em JSON
    metadata = {
        "exp_id": exp_id,
        "branch_name": branch_name,
        "exp_name": experiment_name,
        "exp_dir": str(exp_path),
        "created_at": timestamp,
        "description": description,
        "config_files": [p.name for p in resolved_paths.values()]
    }

    metadata_path = subdirs["logs"] / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print("✓ Criado: logs/metadata.json")

    print("✅ EXPERIMENTO CRIADO COM SUCESSO")
    print(f"📁 Path: {exp_path}")

    return exp_path

def load_experiment(
    experiment_name: str,
    base_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Carrega paths de um experimento existente pelo nome.

    Args:
        experiment_name: Nome do experimento
        base_dir: Diretório base onde estão os experimentos (opcional)

    Returns:
        Dicionário com paths organizados

    Raises:
        FileNotFoundError: Se o experimento não existir
    """
    exp_path = get_experiment_path(experiment_name, base_dir)

    print(f"\n📂 Carregando experimento: {exp_path.name}")

    # Construir dicionário de paths
    paths = {
        "root": exp_path,
        "config": exp_path / "config",
        "model": exp_path / "model",
        "logs": exp_path / "logs",
        "test": exp_path / "predictions_test",
        "test_raw": exp_path / "predictions_test" / "raw",
        "test_metrics": exp_path / "predictions_test" / "metrics",
        "test_plots": exp_path / "predictions_test" / "plots",
        "operational": exp_path / "predictions_operational",
        "operational_raw": exp_path / "predictions_operational" / "raw",
        "operational_plots": exp_path / "predictions_operational" / "plots",
    }

    # Carregar metadata se existir
    metadata_path = paths["logs"] / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"  ID: {metadata.get('exp_id', 'N/A')}")
        print(f"  Name: {metadata.get('exp_name', 'N/A')}")
        print(f"  Created: {metadata.get('created_at', 'N/A')}")

    print("✅ Experimento carregado\n")

    return paths


def save_model(
    model: torch.nn.Module,
    experiment_name: str,
    dataset: Optional[Any] = None,
    model_config: Optional[Dict[str, Any]] = None,
    training_info: Optional[Dict[str, Any]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    filename: str = "model.pth",
    base_dir: Optional[Path] = None
) -> Path:
    """
    Salva o modelo treinado com todos os metadados necessários.

    Args:
        model: Modelo PyTorch treinado
        experiment_name: Nome do experimento
        dataset: Dataset HydroDataset (para extrair scalers e configs automaticamente)
        model_config: Configuração completa da arquitetura do modelo
        training_info: Informações do treinamento (época final, loss, etc.)
        extra_meta: Metadados adicionais opcionais
        filename: Nome do arquivo (padrão: "model.pth")
        base_dir: Diretório base dos experimentos (opcional)

    Returns:
        Path do arquivo salvo

    Examples:
        # Uso completo (RECOMENDADO)
        save_model(
            model=trained_model,
            experiment_name="baseline_lstm",
            dataset=dataset,
            model_config={
                "encoder_input_dim": 66,
                "decoder_input_dim": 55,
                "hidden_dim": 128,
                # ... todos os hiperparâmetros
            },
            training_info={
                "final_epoch": 50,
                "best_val_loss": 0.123,
                "train_loss": 0.098
            }
        )

        # Uso mínimo (compatibilidade com versão antiga)
        save_model(
            model=trained_model,
            experiment_name="baseline_lstm",
            extra_meta={
                "flow_scalers": scalers,
                # ... outros metadados manualmente
            }
        )
    """
    exp_path = get_experiment_path(experiment_name, base_dir)
    model_dir = exp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / filename

    # ==========================================
    # PREPARAR CHECKPOINT COMPLETO
    # ==========================================
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "exp_dir": str(exp_path),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # ==========================================
    # EXTRAIR METADADOS DO DATASET (se fornecido)
    # ==========================================
    if dataset is not None:
        inference_meta = {
            "flow_scalers": dataset.flow_scalers,
            "climate_scalers": dataset.climate_scalers,
            "static_scalers": dataset.static_scalers,
            "stations": dataset.stations,
            "forecast_cols": dataset.forecast_cols,
            "flow_window_config": dataset.flow_window_config,
            "climate_window_config": dataset.climate_window_config,
            "temporal_features": dataset.temporal_features,
            "static_keys": dataset.static_keys,
            "decoder_history": dataset.decoder_history,
            "decoder_horizon": dataset.decoder_horizon,
            "encoder_length": dataset.encoder_length,
            "decoder_length": dataset.decoder_length,
            "forcings": dataset.forcings,
        }
        checkpoint["inference_meta"] = inference_meta
        print("✓ Metadados do dataset extraídos automaticamente")

    # ==========================================
    # ADICIONAR CONFIGURAÇÃO DO MODELO
    # ==========================================
    if model_config is not None:
        checkpoint["model_config"] = model_config
        print("✓ Configuração do modelo salva")

    # ==========================================
    # ADICIONAR INFORMAÇÕES DE TREINAMENTO
    # ==========================================
    if training_info is not None:
        checkpoint["training_info"] = training_info
        print("✓ Informações de treinamento salvas")

    # ==========================================
    # ADICIONAR METADADOS EXTRAS
    # ==========================================
    if extra_meta is not None:
        checkpoint["extra_meta"] = extra_meta
        print("✓ Metadados extras salvos")

    # ==========================================
    # SALVAR
    # ==========================================
    torch.save(checkpoint, model_path)

    # Resumo do que foi salvo
    print(f"\n💾 Modelo salvo: {exp_path.name}/model/{filename}")
    print("📦 Checkpoint contém:")
    print("  - model_state_dict")
    if "inference_meta" in checkpoint:
        print("  - inference_meta (scalers, configs, etc.)")
    if "model_config" in checkpoint:
        print("  - model_config (arquitetura completa)")
    if "training_info" in checkpoint:
        print("  - training_info (histórico de treinamento)")
    if "extra_meta" in checkpoint:
        print("  - extra_meta (metadados adicionais)")

    return model_path

def save_predictions(
    df: pd.DataFrame,
    experiment_name: str,
    mode: str = "test",
    filename: str = "predictions.csv",
    base_dir: Optional[Path] = None
) -> Path:
    """
    Salva previsões em formato CSV.

    Args:
        df: DataFrame com previsões
        experiment_name: Nome do experimento
        mode: "test" ou "operational"
        filename: Nome do arquivo (padrão: "predictions.csv")
        base_dir: Diretório base dos experimentos (opcional)

    Returns:
        Path do arquivo salvo
    """
    if mode not in ["test", "operational"]:
        raise ValueError(f"mode deve ser 'test' ou 'operational', recebido: {mode}")

    exp_path = get_experiment_path(experiment_name, base_dir)

    # Selecionar diretório correto
    if mode == "test":
        save_dir = exp_path / "predictions_test" / "raw"
    else:
        save_dir = exp_path / "predictions_operational" / "raw"

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename

    # Salvar em CSV
    df.to_csv(save_path, index=False, float_format="%.6f")

    print(f"💾 Previsões salvas ({mode}): {exp_path.name}/predictions_{mode}/raw/{filename}")

    return save_path

def save_metrics(
    metrics: Dict[str, Any],
    experiment_name: str,
    filename_base: str = "metrics",
    base_dir: Optional[Path] = None,
    save_json: bool = True,
    save_csv: bool = True,
) -> Dict[str, Path]:
    """
    Salva métricas em JSON e múltiplos arquivos CSV organizados por tipo de evento.
    
    Suporta duas estruturas de entrada:
    1. Estrutura simples: {station: metrics} (métricas gerais por estação)
    2. Estrutura por evento: {event_type: {station: metrics}} (métricas por tipo de evento)

    Args:
        metrics: Dicionário de métricas
        experiment_name: Nome do experimento
        filename_base: Nome base dos arquivos (sem extensão)
        base_dir: Diretório base dos experimentos (opcional)
        save_json: Se True, salva JSON completo (padrão: True)
        save_csv: Se True, salva CSVs organizados (padrão: True)

    Returns:
        Dicionário com paths dos arquivos salvos
    """
    exp_path = get_experiment_path(experiment_name, base_dir)
    metrics_dir = exp_path / "predictions_test" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    
    # Detectar estrutura das métricas
    first_key = next(iter(metrics.keys()), None)
    is_event_based = first_key in ['extreme', 'moderate', 'normal', 'extreme_high', 
                                    'extreme_low', 'moderate_high', 'moderate_low']
    
    if is_event_based:
        event_types = list(metrics.keys())
    else:
        event_types = ['overall']
        metrics = {'overall': metrics}

    # Salvar JSON completo
    if save_json:
        json_path = metrics_dir / f"{filename_base}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        saved_paths["json"] = json_path
        print(f"💾 Métricas salvas (JSON): {exp_path.relative_to(get_experiments_base_dir().parent)}/predictions_test/metrics/{filename_base}.json")

    # Salvar CSVs
    if save_csv:
        # Overall por evento
        rows_overall = []
        for event_type in event_types:
            if event_type not in metrics:
                continue
            station_metrics = metrics[event_type]
            
            for station, m in station_metrics.items():
                if not isinstance(m, dict) or 'overall' not in m:
                    continue
                overall = m['overall']
                row = {
                    'event_type': event_type,
                    'station': station,
                    'rmse': overall.get('rmse'),
                    'mae': overall.get('mae'),
                    'mape': overall.get('mape'),
                    'r2': overall.get('r2'),
                    'nse': overall.get('nse'),
                    'kge': overall.get('kge'),
                    'skill_rmse': overall.get('skill_rmse'),
                    'n_windows': m.get('n_windows'),
                }
                rows_overall.append(row)
        
        if rows_overall:
            df_overall = pd.DataFrame(rows_overall)
            csv_overall_path = metrics_dir / f"{filename_base}_overall.csv"
            df_overall.to_csv(csv_overall_path, index=False, sep=';')
            saved_paths["csv_overall"] = csv_overall_path
            print(f"💾 Métricas salvas (CSV Overall): {csv_overall_path}")

        # Macro por evento
        rows_macro = []
        for event_type in event_types:
            if event_type not in metrics:
                continue
            station_metrics = metrics[event_type]
            
            for station, m in station_metrics.items():
                if not isinstance(m, dict) or 'macro' not in m:
                    continue
                macro = m['macro']
                row = {
                    'event_type': event_type,
                    'station': station,
                    'rmse': macro.get('rmse'),
                    'mae': macro.get('mae'),
                    'mape': macro.get('mape'),
                    'r2': macro.get('r2'),
                    'nse': macro.get('nse'),
                    'n_windows': m.get('n_windows'),
                }
                rows_macro.append(row)
        
        if rows_macro:
            df_macro = pd.DataFrame(rows_macro)
            csv_macro_path = metrics_dir / f"{filename_base}_macro.csv"
            df_macro.to_csv(csv_macro_path, index=False, sep=';')
            saved_paths["csv_macro"] = csv_macro_path
            print(f"💾 Métricas salvas (CSV Macro): {csv_macro_path}")

        # Per horizon por evento
        rows_per_horizon = []
        for event_type in event_types:
            if event_type not in metrics:
                continue
            station_metrics = metrics[event_type]
            
            for station, m in station_metrics.items():
                if not isinstance(m, dict) or 'per_horizon' not in m:
                    continue
                per_horizon = m['per_horizon']
                
                if isinstance(per_horizon, dict):
                    horizons = range(1, len(per_horizon.get('rmse', [])) + 1)
                    for h_idx, h in enumerate(horizons):
                        row = {
                            'event_type': event_type,
                            'station': station,
                            'horizon': h,
                            'rmse': per_horizon.get('rmse', [None])[h_idx],
                            'mae': per_horizon.get('mae', [None])[h_idx],
                            'mape': per_horizon.get('mape', [None])[h_idx],
                            'r2': per_horizon.get('r2', [None])[h_idx],
                            'nse': per_horizon.get('nse', [None])[h_idx],
                        }
                        rows_per_horizon.append(row)
        
        if rows_per_horizon:
            df_per_horizon = pd.DataFrame(rows_per_horizon)
            csv_per_horizon_path = metrics_dir / f"{filename_base}_per_horizon.csv"
            df_per_horizon.to_csv(csv_per_horizon_path, index=False, sep=';')
            saved_paths["csv_per_horizon"] = csv_per_horizon_path
            print(f"💾 Métricas salvas (CSV Per Horizon): {csv_per_horizon_path}")

        # Summary
        rows_summary = []
        for event_type in event_types:
            if event_type not in metrics:
                continue
            station_metrics = metrics[event_type]
            
            total_windows = sum(m.get('n_windows', 0) for m in station_metrics.values() if isinstance(m, dict))
            
            all_overall = [m['overall'] for m in station_metrics.values() if isinstance(m, dict) and 'overall' in m]
            if all_overall:
                row = {
                    'event_type': event_type,
                    'n_stations': len(all_overall),
                    'total_windows': total_windows,
                    'rmse_mean': np.mean([o.get('rmse') for o in all_overall if o.get('rmse') is not None]),
                    'rmse_std': np.std([o.get('rmse') for o in all_overall if o.get('rmse') is not None]),
                    'mae_mean': np.mean([o.get('mae') for o in all_overall if o.get('mae') is not None]),
                    'nse_mean': np.mean([o.get('nse') for o in all_overall if o.get('nse') is not None]),
                }
                rows_summary.append(row)
        
        if rows_summary:
            df_summary = pd.DataFrame(rows_summary)
            csv_summary_path = metrics_dir / f"{filename_base}_summary.csv"
            df_summary.to_csv(csv_summary_path, index=False, sep=';')
            saved_paths["csv_summary"] = csv_summary_path
            print(f"💾 Métricas salvas (CSV Summary): {csv_summary_path}")

    return saved_paths


def save_plot(
    fig: plt.Figure,
    experiment_name: str,
    mode: str = "test",
    name: str = "plot.png",
    dpi: int = 300,
    base_dir: Optional[Path] = None
) -> Path:
    """
    Salva figura matplotlib.

    Args:
        fig: Figura matplotlib
        experiment_name: Nome do experimento
        mode: "test" ou "operational"
        name: Nome do arquivo
        dpi: Resolução da imagem
        base_dir: Diretório base dos experimentos (opcional)

    Returns:
        Path do arquivo salvo
    """
    if mode not in ["test", "operational"]:
        raise ValueError(f"mode deve ser 'test' ou 'operational', recebido: {mode}")

    exp_path = get_experiment_path(experiment_name, base_dir)

    # Selecionar diretório correto
    if mode == "test":
        save_dir = exp_path / "predictions_test" / "plots"
    else:
        save_dir = exp_path / "predictions_operational" / "plots"

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / name

    # Salvar
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    print(f"💾 Plot salvo ({mode}): {exp_path.name}/predictions_{mode}/plots/{name}")

    return save_path


def list_experiments(base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Lista todos os experimentos existentes.

    Args:
        base_dir: Diretório base dos experimentos (opcional)

    Returns:
        Lista de dicionários com informações dos experimentos
    """
    if base_dir is None:
        base_dir = get_experiments_base_dir()

    if not base_dir.exists():
        print(f"⚠️  Diretório não encontrado: {base_dir}")
        return []

    experiments = []

    for exp_dir in sorted(base_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
            continue

        # Carregar metadata se existir
        metadata_path = exp_dir / "logs" / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            experiments.append(metadata)
        else:
            # Criar metadata básico
            experiments.append({
                "exp_dir": str(exp_dir),
                "exp_name": exp_dir.name,
                "created_at": "Unknown"
            })

    return experiments


def print_experiment_summary(
    experiment_name: str,
    base_dir: Optional[Path] = None
) -> None:
    """
    Imprime resumo de um experimento.

    Args:
        experiment_name: Nome do experimento
        base_dir: Diretório base dos experimentos (opcional)
    """
    exp_path = get_experiment_path(experiment_name, base_dir)

    print("\n" + "="*80)
    print("RESUMO DO EXPERIMENTO")
    print("="*80)

    # Carregar metadata
    metadata_path = exp_path / "logs" / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        print("\n📌 Informações:")
        print(f"  ID: {metadata.get('exp_id', 'N/A')}")
        print(f"  Nome: {metadata.get('exp_name', 'N/A')}")
        print(f"  Criado em: {metadata.get('created_at', 'N/A')}")
        print(f"  Descrição: {metadata.get('description', 'N/A')}")

    # Listar arquivos importantes
    print("\n📁 Estrutura:")

    # Modelo
    model_dir = exp_path / "model"
    model_files = list(model_dir.glob("*.pth"))
    if model_files:
        print(f"  ✅ Modelo: {len(model_files)} arquivo(s)")
        for f in model_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    - {f.name} ({size_mb:.2f} MB)")
    else:
        print("  ❌ Modelo: nenhum arquivo")

    # Previsões de teste
    test_raw = exp_path / "predictions_test" / "raw"
    test_files = list(test_raw.glob("*.csv"))
    if test_files:
        print(f"  ✅ Previsões (teste): {len(test_files)} arquivo(s)")
    else:
        print("  ❌ Previsões (teste): nenhum arquivo")

    # Métricas
    metrics_dir = exp_path / "predictions_test" / "metrics"
    metrics_files = list(metrics_dir.glob("*.json"))
    if metrics_files:
        print(f"  ✅ Métricas: {len(metrics_files)} arquivo(s)")
    else:
        print("  ❌ Métricas: nenhum arquivo")

    # Plots
    test_plots = exp_path / "predictions_test" / "plots"
    plot_files = list(test_plots.glob("*.png"))
    if plot_files:
        print(f"  ✅ Plots (teste): {len(plot_files)} arquivo(s)")
    else:
        print("  ❌ Plots (teste): nenhum arquivo")
