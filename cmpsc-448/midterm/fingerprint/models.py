"""Small text CNN and bidirectional LSTM with explicit padding handling."""

import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, architecture, vocab_size, embedding_dim=64, hidden=64):
        super().__init__()
        if architecture not in ('cnn', 'lstm'):
            raise ValueError(f'Unknown architecture: {architecture}')
        self.architecture = architecture
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.3)
        if architecture == 'cnn':
            self.convolutions = nn.ModuleList([
                nn.Conv1d(embedding_dim, hidden, width, padding=width // 2)
                for width in (3, 5, 7)
            ])
            feature_dim = hidden * 3
        else:
            self.forward_lstm = nn.LSTM(embedding_dim, hidden, batch_first=True)
            self.backward_lstm = nn.LSTM(embedding_dim, hidden, batch_first=True)
            feature_dim = hidden * 2
        self.output = nn.Linear(feature_dim, 3)

    def forward(self, tokens, lengths):
        embeddings = self.embedding(tokens)
        if self.architecture == 'cnn':
            # Windows centered on padding must never win the maximum.
            valid = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < lengths.to(tokens.device)[:, None]
            features = [
                torch.relu(layer(embeddings.transpose(1, 2))).masked_fill(~valid[:, None, :], -torch.inf).amax(dim=2)
                for layer in self.convolutions
            ]
            representation = torch.cat(features, dim=1)
        else:
            # Reverse each valid prefix, not the batch's padded width. Two
            # dense one-way passes equal a packed bidirectional LSTM at the
            # last valid states, while avoiding slow packed backward kernels.
            positions = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
            reverse_indices = (lengths.to(tokens.device)[:, None] - 1 - positions).clamp(min=0)
            reversed_embeddings = embeddings.gather(1, reverse_indices[:, :, None].expand_as(embeddings))
            forward, _ = self.forward_lstm(embeddings)
            backward, _ = self.backward_lstm(reversed_embeddings)
            batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
            last = lengths.to(tokens.device) - 1
            representation = torch.cat([forward[batch_indices, last], backward[batch_indices, last]], dim=1)
        return self.output(self.dropout(representation))
