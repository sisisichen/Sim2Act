from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .model import Sim2ActViewRanker


@dataclass(frozen=True)
class FrozenArchitecture:
    global_dim: int = 16
    candidate_dim: int = 24
    global_hidden: int = 64
    candidate_hidden: int = 96
    scoring_hidden_1: int = 128
    scoring_hidden_2: int = 64
    dropout: float = 0.1
    activation: str = "GELU"
    listwise_weight: float = 1.0
    pairwise_weight: float = 0.5
    regression_weight: float = 0.1
    target_temperature: float = 0.05
    ranking_epsilon: float = 1e-6


FROZEN_ARCHITECTURE = FrozenArchitecture()
MODEL_VARIANTS = ("LinearRanker", "LocalMLPRanker", "Sim2ActViewRanker")
MODEL_SEEDS = (2025, 2026, 2027, 2028, 2029)
ANCHOR_SEED = 2025


class LinearRanker(nn.Module):
    """Shared affine scorer over the frozen global and candidate features."""

    def __init__(self, global_dim: int = 16, candidate_dim: int = 24):
        super().__init__()
        self.global_dim = global_dim
        self.candidate_dim = candidate_dim
        self.scorer = nn.Linear(global_dim + candidate_dim, 1)

    def forward(self, global_features: torch.Tensor, candidate_features: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        _validate_inputs(global_features, candidate_features, mask)
        repeated = global_features.unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
        scores = self.scorer(torch.cat((repeated, candidate_features), dim=-1)).squeeze(-1)
        return scores.masked_fill(~mask, float("-inf"))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)


class LocalMLPRanker(nn.Module):
    """Frozen STEP 5A encoders without mean/max candidate-set context."""

    def __init__(self, global_dim: int = 16, candidate_dim: int = 24,
                 global_hidden: int = 64, candidate_hidden: int = 96):
        super().__init__()
        self.global_dim = global_dim
        self.candidate_dim = candidate_dim
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, global_hidden), nn.LayerNorm(global_hidden), nn.GELU(),
            nn.Linear(global_hidden, global_hidden), nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, candidate_hidden), nn.LayerNorm(candidate_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(candidate_hidden, candidate_hidden), nn.GELU(),
        )
        self.scoring_head = nn.Sequential(
            nn.Linear(global_hidden + candidate_hidden, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, global_features: torch.Tensor, candidate_features: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        _validate_inputs(global_features, candidate_features, mask)
        global_embedding = self.global_encoder(global_features)
        candidate_embedding = self.candidate_encoder(candidate_features)
        repeated = global_embedding.unsqueeze(1).expand(-1, candidate_embedding.shape[1], -1)
        scores = self.scoring_head(torch.cat((repeated, candidate_embedding), dim=-1)).squeeze(-1)
        return scores.masked_fill(~mask, float("-inf"))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)


def _validate_inputs(global_features: torch.Tensor, candidate_features: torch.Tensor,
                     mask: torch.Tensor) -> None:
    if global_features.ndim != 2 or candidate_features.ndim != 3:
        raise ValueError("ranker input shape mismatch")
    if mask.shape != candidate_features.shape[:2] or not torch.all(mask.any(dim=1)):
        raise ValueError("invalid candidate mask")


def make_model(variant: str, global_dim: int = 16, candidate_dim: int = 24) -> nn.Module:
    if variant == "LinearRanker":
        return LinearRanker(global_dim, candidate_dim)
    if variant == "LocalMLPRanker":
        return LocalMLPRanker(global_dim, candidate_dim)
    if variant == "Sim2ActViewRanker":
        return Sim2ActViewRanker(global_dim, candidate_dim)
    raise ValueError(f"unknown model variant: {variant}")


def architecture_manifest(variant: str) -> dict:
    model = make_model(variant)
    return {
        "variant": variant,
        "frozen_step5a_architecture": asdict(FROZEN_ARCHITECTURE),
        "uses_candidate_set_mean_context": variant == "Sim2ActViewRanker",
        "uses_candidate_set_max_context": variant == "Sim2ActViewRanker",
        "candidate_order_invariant": True,
        "parameter_count": model.trainable_parameter_count,
    }
