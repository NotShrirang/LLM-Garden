import torch
import torch.nn as nn
import math
from config import MistralConfig
from layers import RMSNormLayer, MistralBlock
    

class Mistral(nn.Module):
    def __init__(self, args: MistralConfig):
        super().__init__()
        self.embed_dim = args.dim
        self.num_layers = args.n_layers

        self.embedding = nn.Embedding(args.vocab_size, self.embed_dim)
        self.layers = nn.ModuleList([MistralBlock(args) for _ in range(self.num_layers)])
        self.norm = RMSNormLayer(args)
        self.output_layer = nn.Linear(self.embed_dim, args.vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output_layer(x)
        return x

if __name__ == '__main__':
    args = MistralConfig()
    model = Mistral(args)
    print(f"Mistral - {args.n_layers} layers loaded with {sum(p.numel() for p in model.parameters()) / 1e6} M parameters")

    tokens = torch.randint(0, args.vocab_size, (args.max_batch_size, 1))
    logits = model(tokens)
    print(logits.shape)