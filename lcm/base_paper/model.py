import torch.nn as nn

from lcm.base_paper.layers import PreNet, PostNet, TransformerDecoder
  

class BaseLCM(nn.Module):
    """
    Base Large Concept Model
    - PreNet: Maps input embeddings to hidden dim
    - TransformerDecoder: Decoder-only Transformer
    - PostNet: Maps hidden dim to embedding space
    """

    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, ff_dim, output_dim):
        super(BaseLCM, self).__init__()
        self.prenet = PreNet(input_dim, hidden_dim)
        self.decoder = TransformerDecoder(hidden_dim, num_heads, num_layers, ff_dim)
        self.postnet = PostNet(hidden_dim, output_dim)

    def forward(self, x):
        x = self.prenet(x)
        x = self.decoder(x)
        x = self.postnet(x)
        return x
