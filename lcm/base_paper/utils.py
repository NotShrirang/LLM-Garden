import torch
import torch.nn.functional as F
import os


def compute_accuracy(predicted, target, threshold=0.1):
    cos_sim = F.cosine_similarity(predicted, target, dim=-1)
    correct = (cos_sim > threshold).float()
    accuracy = correct.mean().item()
    return accuracy


def load_glove_embeddings(file_path: str, vocab_size=5000):
    """Load GloVe embeddings from a file for a small subset."""
    embeddings = {}
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= vocab_size:
                break
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(x) for x in values[1:]], dtype=torch.float)
            embeddings[word] = vector
    return embeddings


def prepare_embeddings(glove_embeddings, batch_size, sequence_length, dim):
    """Randomly sample embeddings from the loaded GloVe vectors."""
    selected_vectors = torch.stack(
        [glove_embeddings[word] for word in list(glove_embeddings.keys())[:sequence_length]]
    )
    input_embeddings = selected_vectors.unsqueeze(0).repeat(batch_size, 1, 1)
    return input_embeddings
