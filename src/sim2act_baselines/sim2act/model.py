from __future__ import annotations

import torch
from torch import nn


class Sim2ActViewRanker(nn.Module):
    def __init__(self, global_dim: int, candidate_dim: int, global_hidden: int = 64, candidate_hidden: int = 96):
        super().__init__()
        self.global_dim = global_dim; self.candidate_dim = candidate_dim
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, global_hidden), nn.LayerNorm(global_hidden), nn.GELU(),
            nn.Linear(global_hidden, global_hidden), nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, candidate_hidden), nn.LayerNorm(candidate_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(candidate_hidden, candidate_hidden), nn.GELU(),
        )
        combined = candidate_hidden + global_hidden + candidate_hidden * 2
        self.scoring_head = nn.Sequential(
            nn.Linear(combined, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, global_features: torch.Tensor, candidate_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if global_features.ndim != 2 or candidate_features.ndim != 3 or mask.shape != candidate_features.shape[:2]:
            raise ValueError("ranker input shape mismatch")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("each state needs at least one candidate")
        global_embedding = self.global_encoder(global_features)
        candidate_embedding = self.candidate_encoder(candidate_features)
        valid = mask.unsqueeze(-1)
        mean = (candidate_embedding * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        maximum = candidate_embedding.masked_fill(~valid, torch.finfo(candidate_embedding.dtype).min).max(dim=1).values
        context = torch.cat((mean, maximum), dim=-1)
        repeated_global = global_embedding.unsqueeze(1).expand(-1, candidate_embedding.shape[1], -1)
        repeated_context = context.unsqueeze(1).expand(-1, candidate_embedding.shape[1], -1)
        scores = self.scoring_head(torch.cat((candidate_embedding, repeated_global, repeated_context), dim=-1)).squeeze(-1)
        return scores.masked_fill(~mask, float("-inf"))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
