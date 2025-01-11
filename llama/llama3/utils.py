import torch


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 500000.0) -> torch.Tensor:
    assert head_dim % 2 == 0, "head_dim must be divisible by 2"

    theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    m = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in polar form c = r * exp(i * m * theta)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embedding(x: torch.Tensor, freqs: torch.Tensor, device) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )