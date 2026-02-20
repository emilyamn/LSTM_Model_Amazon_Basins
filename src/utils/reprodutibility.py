"""
Garantir reprodutibilidade ao inicializar a rede com pesos idênticos
"""

import random
import os
import torch
import numpy as np

def set_seed(seed: int = 42):
    """Fixa as sementes para garantir reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🔒 Seed fixada em: {seed}")
