from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class Llama3Config:
    dim: int = 1024
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = 4
    vocab_size: int = 128256
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 8192

    device: str = None
