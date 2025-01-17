import dataclasses

@dataclasses.dataclass
class BaseLCMConfig:
    batch_size = 4
    seq_len = 10
    input_dim = 300
    hidden_dim = 2048
    num_heads = 8
    num_layers = 16
    ff_dim = 1024
    output_dim = 300
    threshold = 0.1