from dataclasses import dataclass
from typing import Optional


@dataclass
class MistralConfig:
    dim: int = 4096
    n_layers: int = 32
    head_dim: int = 128
    hidden_dim: int = 14336
    n_heads: int = 32
    n_kv_heads: Optional[int] = 8
    window_size: int = 4096
    context_length: int = 8192
    vocab_size: int = 32000
    
    norm_eps: float = 1e-5
    max_batch_size: int = 32

    device: str = None
