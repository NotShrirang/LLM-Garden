import torch
import torch.nn as nn

from config import Llama3Config
from layers import RMSNorm, EncoderBlock
from utils import precompute_theta_pos_frequencies


class Llama3(nn.Module):

    def __init__(self, args: Llama3Config):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        h = self.tok_embeddings(tokens)

        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        logits = self.output(h).float()
        return logits


if __name__ == '__main__':
    args = Llama3Config()
    model = Llama3(args)
    print(f"Llama 3 - 8B loaded with {sum(p.numel() for p in model.parameters()) / 1e6} M parameters")

    # Test the model
    tokens = torch.randint(0, args.vocab_size, (args.max_batch_size, 1))
    logits = model(tokens, 0)
    print(logits.shape)
