import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Llama2Config
from utils import apply_rotary_embedding, repeat_kv


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x)


class SelfAttention(nn.Module):
    def __init__(self, args: Llama2Config) -> None:
        super().__init__()

        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads

        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads *
                            self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads *
                            self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.register_buffer("cache_k", torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)), persistent=False)
        self.register_buffer("cache_v", torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_embedding(xq, freqs_complex, x.device)
        xk = apply_rotary_embedding(xk, freqs_complex, x.device)
        xv = apply_rotary_embedding(xv, freqs_complex, x.device)

        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        keys = self.cache_k[:batch_size, :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(
            xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_len = keys.size(2)
        mask = torch.triu(
            torch.full((seq_len, attn_len), float('-inf'), device=x.device),
            diagonal=1
        )
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(scores, values)
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1))

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: Llama2Config) -> None:
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * args.dim)

        hidden = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = F.silu(self.w1(x))
        return self.w2(swish) + self.w3(x)


class DecoderBlock(nn.Module):
    def __init__(self, args: Llama2Config) -> None:
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
