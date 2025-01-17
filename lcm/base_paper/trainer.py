import torch

from lcm.base_paper.config import BaseLCMConfig
from lcm.base_paper.model import BaseLCM
from lcm.base_paper.utils import compute_accuracy, load_glove_embeddings, prepare_embeddings



def train_lcm(config: BaseLCMConfig, glove_file: str, epochs: int, threshold: float, learning_rate: float = 1e-4):
    batch_size = config.batch_size
    seq_len = config.seq_len
    input_dim = config.input_dim
    hidden_dim = config.hidden_dim
    num_heads = config.num_heads
    num_layers = config.num_layers
    ff_dim = config.ff_dim
    output_dim = config.output_dim

    glove_embeddings = load_glove_embeddings(glove_file, vocab_size=10000)
    input_embeddings = prepare_embeddings(glove_embeddings, batch_size, seq_len, dim=input_dim)

    print(f"Input embeddings shape: {input_embeddings.shape}")

    model = BaseLCM(
        input_dim = input_dim,
        hidden_dim = hidden_dim,
        num_heads = num_heads,
        num_layers=num_layers,
        ff_dim = ff_dim,
        output_dim = output_dim
    )

    print(f"Model with {sum(p.numel() for p in model.parameters()) / 1e6} M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), eps=1e-5, weight_decay=0.1)
    criterion = torch.nn.MSELoss()

    target_embeddings = input_embeddings + torch.randn_like(input_embeddings) * 0.01

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output_embeddings = model(input_embeddings)
        loss = criterion(output_embeddings, input_embeddings)
        loss.backward()
        optimizer.step()

        accuracy = compute_accuracy(output_embeddings, target_embeddings, threshold)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {accuracy * 100:.2f}%")
    
    return model


if __name__ == '__main__':
    config = BaseLCMConfig()
    model = train_lcm(config, glove_file='glove.6B.300d.txt', epochs=10, threshold=0.1)
    print("Training completed.")