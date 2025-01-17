import torch
import torch.nn as nn


class PreNet(nn.Module):
    """This maps model's input embeddings to models hidden dim after normalization"""
    def __init__(self, input_size, hidden_size):
        super(PreNet, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.scaler_mean = 0.0
        self.scaler_std = 1.0

    def normalize(self, x):
        return (x - self.scaler_mean) / self.scaler_std

    def forward(self, x):
        x = self.normalize(x)
        x = self.linear(x)
        return x
  

class PostNet(nn.Module):
    """This maps model's hidden dim to embedding space after de-normalization"""
    def __init__(self, hidden_size, output_size):
        super(PostNet, self).__init__()
        self.linear = nn.Linear(hidden_size, output_size)
        self.scaler_mean = 0.0
        self.scaler_std = 1.0

    def denormalize(self, x):
        return x * self.scaler_std + self.scaler_mean

    def forward(self, x):
        x = self.linear(x)
        x = self.denormalize(x)
        return x
  

class TransformerDecoder(nn.Module):
    """This is a standard Decoder-only Transformer"""
    def __init__(self, hidden_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.pos_encoder = nn.Parameter(torch.zeros(1, 512, hidden_dim))

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len]
        for layer in self.layers:
            x = layer(x, x)
        return x
