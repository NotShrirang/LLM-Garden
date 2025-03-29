import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import MistralConfig


class FeedForward(nn.Module):
    def __init__(self, args: MistralConfig):
        super().__init__()
        self.dim = args.dim
        self.hidden_dim = args.hidden_dim

        self.linear1 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.linear2 = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.linear3 = nn.Linear(self.dim, self.hidden_dim, bias=False)

    def forward(self, x):
        output = self.linear1(x)
        output = F.silu(output)
        output = output * self.linear3(x)
        output = self.linear2(output)
        return output
    

class RMSNormLayer(nn.Module):
    def __init__(self, args: MistralConfig):
        super().__init__()
        self.dim = args.dim
        self.eps = args.norm_eps

        self.gamma = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps) * self.gamma


class SlidingWindowAttentionGQA(nn.Module):
    def __init__(self, embed_size, window_size, num_heads, num_query_heads, n_kv_heads):
        """
        embed_size: total embedding dimension.
        window_size: radius of the sliding window.
        num_heads: total number of query heads.
        num_query_heads: number of query heads (GQA uses fewer query heads than num_heads).
        n_kv_heads: number of key-value heads (typically equal to num_query_heads in GQA).
        """
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        assert num_query_heads <= num_heads, "num_query_heads must be <= num_heads"
        assert n_kv_heads <= num_heads, "n_kv_heads must be <= num_heads"
        
        self.embed_size = embed_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.num_query_heads = num_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = embed_size // num_heads
        self.group_size = n_kv_heads // num_query_heads

        self.W_q = nn.Linear(embed_size, num_query_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(embed_size, n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(embed_size, n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(num_query_heads * self.head_dim, embed_size, bias=False)
        
    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_length, embed_size]
        """
        batch_size, seq_length, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_length, self.num_query_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_length, self.n_kv_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_length, self.n_kv_heads, self.head_dim)

        queries_per_kv = self.num_query_heads // self.n_kv_heads
        
        outputs = []
        for i in range(seq_length):
            start = max(0, i - self.window_size)
            end = min(seq_length, i + self.window_size + 1)

            Q_i = Q[:, i:i+1, :, :]

            K_window = K[:, start:end, :, :]
            V_window = V[:, start:end, :, :]

            K_window_expanded = K_window.repeat_interleave(queries_per_kv, dim=2)
            V_window_expanded = V_window.repeat_interleave(queries_per_kv, dim=2)

            K_window_expanded = K_window_expanded.permute(0, 2, 1, 3)
            V_window_expanded = V_window_expanded.permute(0, 2, 1, 3)
            Q_i = Q_i.permute(0, 2, 1, 3)

            scores = torch.matmul(Q_i, K_window_expanded.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = torch.softmax(scores, dim=-1)

            out = torch.matmul(attn, V_window_expanded)

            out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, 1, self.num_query_heads * self.head_dim)
            outputs.append(out)

        output = torch.cat(outputs, dim=1)

        return self.W_o(output)


class MistralBlock(nn.Module):
    def __init__(self, args: MistralConfig):
        super().__init__()
        self.attention = SlidingWindowAttentionGQA(args.dim, args.window_size, args.n_heads, args.n_heads, args.n_kv_heads)
        self.ffn = FeedForward(args)
        self.norm1 = RMSNormLayer(args)
        self.norm2 = RMSNormLayer(args)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
