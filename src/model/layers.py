"""
Camadas personalizadas para o modelo.
"""
import torch
import torch.nn as nn


class StaticEmbedding(nn.Module):
    """Camada de embedding para features estáticas."""
    
    def __init__(self, n_static: int, hidden_dim: int):
        """
        Inicializa a camada de embedding estático.
        
        Args:
            n_static: Dimensão das features estáticas
            hidden_dim: Dimensão do embedding
        """
        super().__init__()
        self.proj = nn.Linear(n_static, hidden_dim)

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        """
        Aplica projeção linear às features estáticas.
        
        Args:
            static: Tensor de features estáticas
            
        Returns:
            Tensor embeddado
        """
        return self.proj(static)