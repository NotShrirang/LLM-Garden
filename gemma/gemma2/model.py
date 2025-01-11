import torch
import torch.nn as nn
from typing import Optional, Tuple

from config import Gemma2Config
from layers import GemmaRMSNorm, GemmaDecoderLayer
from utils import KVCache


class Gemma2Model(nn.Module):
    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embedding
    
    def forward(
        self,
        idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        hidden_states = self.embedding(idx)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
        
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Gemma2ForCausalLM(nn.Module):
    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.config = config
        self.model = Gemma2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embedding
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embedding.weight

    def forward(
        self,
        idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        outputs: torch.FloatTensor = self.model(
            idx=idx,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        
        logits = outputs
        logits: torch.FloatTensor = self.lm_head(logits)
        logits = logits.float()

        return_data = {
            "logits": logits
        }

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache
        
        return return_data
    
    def _sample_top_p(self, logits, top_p=0.9, temperature=1.0):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
        probs = nn.functional.softmax(logits / temperature, dim=-1)
        return probs
    
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
    ) -> Tuple:
        kv_cache = KVCache()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        idx = idx.to(device)

        for _ in range(max_new_tokens):
            outputs = self(
                idx=idx,
                kv_cache=kv_cache,
            )
            logits = outputs["logits"][:, -1, :]
            logits = logits / temperature
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1)

            if top_p < 1.0:
                next_token = self._sample_top_p(next_token, logits, top_p)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            kv_cache = outputs["kv_cache"]

        return input_ids


if __name__ == '__main__':
    args = Gemma2Config(
        vocab_size=256000,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
    )
    model = Gemma2ForCausalLM(args)
    print(f"Gemma 2 - 2B loaded with {sum(p.numel() for p in model.parameters()) / 1e6} M parameters")

    # Test the model
    tokens = torch.randint(0, args.vocab_size, (args.hidden_size, 1))
    logits = model.generate(tokens, 10)
    print(logits.shape)