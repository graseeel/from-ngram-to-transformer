from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ngram_transformer.config import TransformerConfig
from ngram_transformer.ml.sampling import sample_from_logits


def build_causal_mask(block_size: int, device: torch.device | str) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    # The lower-triangular mask prevents each position from seeing future tokens.
    return torch.tril(torch.ones((block_size, block_size), dtype=torch.bool, device=device))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        mask = build_causal_mask(config.block_size, "cpu").view(
            1,
            1,
            config.block_size,
            config.block_size,
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_size = x.shape
        q, k, v = self.c_attn(x).split(embedding_size, dim=2)
        q = q.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.get_buffer("causal_mask")[:, :, :sequence_length, :sequence_length]
        scores = scores.masked_fill(~mask, float("-inf"))
        attention = self.dropout(torch.softmax(scores, dim=-1))
        y = attention @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_size)
        return self.dropout(self.c_proj(y))


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.vocab_size is None:
            raise ValueError("TransformerConfig.vocab_size must be set")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.n_layer)))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length = idx.shape
        if sequence_length > self.config.block_size:
            raise ValueError("sequence length exceeds configured block_size")
        positions = torch.arange(sequence_length, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(batch_size * sequence_length, -1), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        seed_ids: list[int],
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        greedy: bool = False,
        device: str | torch.device = "cpu",
    ) -> list[int]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.eval()
        generator = torch.Generator(device=str(device))
        if seed is not None:
            generator.manual_seed(seed)
        ids = torch.tensor([seed_ids], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            idx_cond = ids[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            next_id = sample_from_logits(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
                greedy=greedy,
            )
            ids = torch.cat((ids, next_id[:, None]), dim=1)
        return ids[0].tolist()
