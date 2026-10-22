"""A randomly initialized, decoder-only character Transformer."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int
    width: int = 128
    heads: int = 4
    layers: int = 3
    context: int = 128
    dropout: float = 0.1


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.projection = nn.Linear(config.width, config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width), nn.GELU(),
            nn.Linear(4 * config.width, config.width), nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        heads = self.config.heads
        q, k, v = [t.reshape(batch, length, heads, width // heads).transpose(1, 2) for t in (q, k, v)]
        # is_causal prevents the model from reading the character it predicts.
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.config.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
        x = x + self.dropout(self.projection(attended))
        return x + self.mlp(self.norm2(x))


class TinyLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.width % config.heads:
            raise ValueError('Embedding width must be divisible by head count')
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        if tokens.shape[1] > self.config.context:
            raise ValueError('Input exceeds model context')
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.tokens(tokens) + self.positions(positions)
        return self.head(self.norm(self.blocks(x)))
