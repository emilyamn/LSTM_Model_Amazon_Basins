"""
Módulo para carregar configurações de arquivos YAML.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pathlib
import yaml


class ConfigLoader:
    """Classe para carregar configurações de arquivos YAML."""

    def __init__(self, config_dir: Optional[pathlib.Path] = None):
        """
        Inicializa o carregador de configurações.

        Args:
            config_dir: Diretório onde estão os arquivos de configuração
        """
        if config_dir is None:
            # Tenta encontrar automaticamente a pasta config
            current_dir = pathlib.Path(__file__).parent
            # Ajuste para sua estrutura: src/utils → src → raiz → config
            self.config_dir = current_dir.parent.parent / "config"
        else:
            self.config_dir = config_dir

        if not self.config_dir.exists():
            raise FileNotFoundError(f"Diretório de configuração não encontrado: {self.config_dir}")

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Carrega um arquivo de configuração YAML.

        Args:
            config_file: Nome do arquivo de configuração (ex: 'data_config.yaml')

        Returns:
            Dicionário com as configurações
        """
        config_path = self.config_dir / config_file

        if not config_path.exists():
            available = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
            available_names = [f.name for f in available]
            raise FileNotFoundError(
                f"Arquivo de configuração não encontrado: {config_path}\n"
                f"Arquivos disponíveis: {available_names}"
            )

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config or {}

    def get_feature_config(self) -> Dict[str, Any]:
        """
        Carrega especificamente as configurações de features.

        Returns:
            Dicionário com configurações de features

        Raises:
            ValueError: Se alguma configuração necessária estiver faltando
        """
        config = self.load_config("data_config.yaml")

        # Verificar se a seção feature_windows existe
        if "feature_windows" not in config:
            raise ValueError(
                "❌ ERRO: Seção 'feature_windows' não encontrada no arquivo data_config.yaml.\n"
                "Por favor, adicione a seção 'feature_windows' com as seguintes chaves:\n"
                "  - precipitation_ma\n"
                "  - precipitation_cum\n"
                "  - forecast_ma (ou use os mesmos valores de precipitation_ma)\n"
                "  - forecast_cum (ou use os mesmos valores de precipitation_cum)\n"
                "  - evapotranspiration_ma\n"
                "  - anomaly_ma\n"
                "  - api_k_list\n\n"
                "Exemplo de estrutura:\n"
                "feature_windows:\n"
                "  precipitation_ma: [3, 7, 15]\n"
                "  precipitation_cum: [3, 5, 7, 10]\n"
                "  # ... outras configurações"
            )

        feature_config = config["feature_windows"]

        # Lista de chaves obrigatórias
        required_keys = [
            "precipitation_ma",
            "precipitation_cum",
            "evapotranspiration_ma",
            "anomaly_ma",
            "api_k_list"
        ]

        # Verificar chaves faltantes
        missing_keys = []
        for key in required_keys:
            if key not in feature_config:
                missing_keys.append(key)

        if missing_keys:
            raise ValueError(
                f"❌ ERRO: Configurações obrigatórias faltando no arquivo data_config.yaml:\n"
                f"Chaves faltantes: {', '.join(missing_keys)}\n\n"
                f"Por favor, adicione estas configurações à seção 'feature_windows'.\n"
                f"Arquivo de configuração: {self.config_dir / 'data_config.yaml'}"
            )

        # Verificar se as configurações são listas válidas
        for key, value in feature_config.items():
            if not isinstance(value, list):
                raise ValueError(
                    f"❌ ERRO: A configuração '{key}' deve ser uma lista, mas recebeu: {type(value)}.\n"
                    f"Valor atual: {value}\n"
                    f"Exemplo correto: {key}: [3, 7, 15]"
                )

        # Configurações opcionais com fallback
        if "forecast_ma" not in feature_config:
            feature_config["forecast_ma"] = feature_config["precipitation_ma"]
            print("⚠️  Aviso: 'forecast_ma' não definido. Usando valores de 'precipitation_ma'")

        if "forecast_cum" not in feature_config:
            feature_config["forecast_cum"] = feature_config["precipitation_cum"]
            print("⚠️  Aviso: 'forecast_cum' não definido. Usando valores de 'precipitation_cum'")

        return feature_config

    def get_split_config(self) -> Dict[str, Any]:
        """
        Carrega as configurações de split (train/val/test).

        Returns:
            Dicionário com configurações de split

        Raises:
            ValueError: Se alguma configuração necessária estiver faltando
        """
        config = self.load_config("data_config.yaml")

        # Verificar se a seção split_config existe
        if "split_config" not in config:
            raise ValueError(
                "❌ ERRO: Seção 'split_config' não encontrada no arquivo data_config.yaml.\n"
                "Por favor, adicione a seção 'split_config' com as seguintes chaves:\n"
                "  - train_ratio (ex: 0.95)\n"
                "  - val_ratio (ex: 0.025)\n"
                "  - test_ratio (ex: 0.025)\n"
                "  - gap (ex: 128)\n"
                "  - window_stride (ex: 1)\n\n"
                "Exemplo de estrutura:\n"
                "split_config:\n"
                "  train_ratio: 0.95\n"
                "  val_ratio: 0.025\n"
                "  test_ratio: 0.025\n"
                "  gap: 128\n"
                "  window_stride: 1\n"
                "  # A soma de train_ratio + val_ratio + test_ratio deve ser 1.0"
            )

        split_config = config["split_config"]

        # Lista de chaves obrigatórias
        required_keys = ["train_ratio", "val_ratio", "test_ratio", "gap", "window_stride"]

        # Verificar chaves faltantes
        missing_keys = []
        for key in required_keys:
            if key not in split_config:
                missing_keys.append(key)

        if missing_keys:
            raise ValueError(
                f"❌ ERRO: Configurações de split obrigatórias faltando no arquivo data_config.yaml:\n"
                f"Chaves faltantes: {', '.join(missing_keys)}\n\n"
                f"Por favor, adicione estas configurações à seção 'split_config'.\n"
                f"Arquivo de configuração: {self.config_dir / 'data_config.yaml'}"
            )

        # Validar que a soma dos ratios é 1.0
        total_ratio = split_config["train_ratio"] + split_config["val_ratio"] + split_config["test_ratio"]
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(
                f"❌ ERRO: A soma de train_ratio + val_ratio + test_ratio deve ser 1.0, mas é {total_ratio:.3f}\n"
                f"Por favor, corrija no arquivo data_config.yaml"
            )

        return split_config

    def calculate_split_dates(self, start_date: str, end_date: str) -> Dict[str, str]:
        """
        Calcula as datas de corte para train/val/test baseado nas configurações.

        Args:
            start_date: Data inicial no formato 'YYYY-MM-DD'
            end_date: Data final no formato 'YYYY-MM-DD'

        Returns:
            Dicionário com as datas de corte
        """
        split_config = self.get_split_config()

        # Converter strings para datetime
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        total_days = (end - start).days

        # Calcular número de dias para cada split
        train_days = int(total_days * split_config["train_ratio"])
        val_days = int(total_days * split_config["val_ratio"])
        # test_days é o restante

        # Calcular datas de corte
        train_end = start + timedelta(days=train_days)
        val_end = train_end + timedelta(days=val_days)

        return {
            "train_end": train_end.strftime("%Y-%m-%d"),
            "val_end": val_end.strftime("%Y-%m-%d"),
            "test_end": end_date,
            "train_start": start_date,
            "total_days": total_days,
            "train_days": train_days,
            "val_days": val_days,
            "test_days": total_days - train_days - val_days
        }

    @staticmethod
    def validate_feature_config(feature_config: Dict[str, Any]) -> bool:
        """
        Valida se as configurações de features são válidas.

        Args:
            feature_config: Dicionário com configurações de features

        Returns:
            True se válido, False se não

        Raises:
            ValueError: Com mensagem detalhada do problema
        """
        errors = []

        # Verificar tipos
        list_keys = ["precipitation_ma", "precipitation_cum", "forecast_ma",
                    "forecast_cum", "evapotranspiration_ma", "anomaly_ma", "api_k_list"]

        for key in list_keys:
            if key in feature_config:
                if not isinstance(feature_config[key], list):
                    errors.append(f"'{key}' deve ser uma lista (tipo atual: {type(feature_config[key])})")
                elif len(feature_config[key]) == 0:
                    errors.append(f"'{key}' não pode ser uma lista vazia")

        # Verificar valores específicos
        if "api_k_list" in feature_config:
            for k in feature_config["api_k_list"]:
                if not 0 < k < 1:
                    errors.append(f"Valores de api_k_list devem estar entre 0 e 1. Valor inválido: {k}")

        if errors:
            error_msg = "\n".join([f"  • {error}" for error in errors])
            raise ValueError(f"❌ ERRO na validação das configurações:\n{error_msg}")

        return True

    @staticmethod
    def validate_split_config(split_config: Dict[str, Any]) -> bool:
        """
        Valida se as configurações de split são válidas.

        Args:
            split_config: Dicionário com configurações de split

        Returns:
            True se válido, False se não

        Raises:
            ValueError: Com mensagem detalhada do problema
        """
        errors = []

        # Verificar tipos
        for key in ["train_ratio", "val_ratio", "test_ratio"]:
            if key in split_config:
                value = split_config[key]
                if not isinstance(value, (int, float)):
                    errors.append(f"'{key}' deve ser um número (tipo atual: {type(value)})")
                elif not 0 <= value <= 1:
                    errors.append(f"'{key}' deve estar entre 0 e 1. Valor atual: {value}")

        for key in ["gap", "window_stride"]:
            if key in split_config:
                value = split_config[key]
                if not isinstance(value, int):
                    errors.append(f"'{key}' deve ser um inteiro (tipo atual: {type(value)})")
                elif value < 1:
                    errors.append(f"'{key}' deve ser maior que 0. Valor atual: {value}")

        # Verificar soma dos ratios
        if all(k in split_config for k in ["train_ratio", "val_ratio", "test_ratio"]):
            total = (split_config["train_ratio"] +
                    split_config["val_ratio"] +
                    split_config["test_ratio"])
            if abs(total - 1.0) > 0.001:
                errors.append(f"A soma train_ratio + val_ratio + test_ratio deve ser 1.0, mas é {total:.3f}")

        if errors:
            error_msg = "\n".join([f"  • {error}" for error in errors])
            raise ValueError(f"❌ ERRO na validação das configurações de split:\n{error_msg}")

        return True

    @staticmethod
    def create_default_config(config_dir: pathlib.Path) -> None:
        """
        Cria um arquivo de configuração padrão se não existir.

        Args:
            config_dir: Diretório onde criar o arquivo
        """
        config_path = config_dir / "data_config.yaml"

        if not config_path.exists():
            default_config = {
                "feature_windows": {
                    "precipitation_ma": [3, 7, 15],
                    "precipitation_cum": [3, 5, 7, 10],
                    "forecast_ma": [3, 7, 15],
                    "forecast_cum": [3, 5, 7, 10],
                    "evapotranspiration_ma": [7, 14, 30],
                    "anomaly_ma": [3, 7],
                    "api_k_list": [0.70, 0.80, 0.85, 0.90, 0.92, 0.95]
                },
                "split_config": {
                    "train_ratio": 0.95,
                    "val_ratio": 0.025,
                    "test_ratio": 0.025,
                    "gap": 128,
                    "window_stride": 1
                },
                "data_paths": {
                    "complete_series_dir": "data/complete_series",
                    "processed_dir": "data/processed",
                    "models_dir": "data/models"
                },
                "model_config": {
                    "train_test_split": 0.8,
                    "validation_split": 0.2,
                    "random_seed": 42
                }
            }

            config_dir.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

            print(f"✅ Arquivo de configuração padrão criado: {config_path}")
            print("Por favor, ajuste os valores conforme necessário.")


def load_feature_config(validate: bool = True) -> Dict[str, Any]:
    """
    Função de conveniência para carregar configurações de features.

    Args:
        validate: Se True, valida as configurações carregadas

    Returns:
        Dicionário com configurações de features
    """
    loader = ConfigLoader()
    config = loader.get_feature_config()

    if validate:
        loader.validate_feature_config(config)

    return config


def load_split_config(validate: bool = True) -> Dict[str, Any]:
    """
    Função de conveniência para carregar configurações de split.

    Args:
        validate: Se True, valida as configurações carregadas

    Returns:
        Dicionário com configurações de split
    """
    loader = ConfigLoader()
    config = loader.get_split_config()

    if validate:
        loader.validate_split_config(config)

    return config

def load_config(config_path: pathlib.Path) -> Dict[str, Any]:
    """
    Função de compatibilidade para carregar qualquer arquivo de configuração.

    Args:
        config_path: Caminho completo para o arquivo de configuração

    Returns:
        Dicionário com as configurações
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config or {}

def load_all_configs(config_dir: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    """
    Carrega todos os arquivos de configuração de um diretório.

    Args:
        config_dir: Diretório contendo os arquivos de configuração

    Returns:
        Dicionário com todas as configurações carregadas
    """
    configs = {}

    # Arquivos comuns de configuração
    config_files = {
        'data': 'data_config.yaml',
        'model': 'model_config.yaml',
        'training': 'training_config.yaml'
    }

    for key, filename in config_files.items():
        filepath = config_dir / filename
        if filepath.exists():
            configs[key] = load_config(filepath)
        else:
            print(f"⚠️  Aviso: Arquivo {filename} não encontrado")

    return configs