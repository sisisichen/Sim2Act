from __future__ import annotations

import torch
import torch.nn.functional as functional


def ranking_loss(scores: torch.Tensor, gains: torch.Tensor, mask: torch.Tensor, target_temperature: float = 0.05,
                 pairwise_epsilon: float = 1e-6) -> dict[str, torch.Tensor]:
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    masked_targets = (gains / target_temperature).masked_fill(~mask, float("-inf"))
    target_distribution = torch.softmax(masked_targets, dim=1)
    log_prediction = torch.log_softmax(masked_scores, dim=1)
    listwise = -(target_distribution * log_prediction).masked_fill(~mask, 0.0).sum(dim=1).mean()
    pair_losses = []
    for batch in range(scores.shape[0]):
        valid_scores = scores[batch][mask[batch]]; valid_gains = gains[batch][mask[batch]]
        difference = valid_gains[:, None] - valid_gains[None, :]
        upper = torch.triu(torch.ones_like(difference, dtype=torch.bool), diagonal=1) & (difference.abs() > pairwise_epsilon)
        if upper.any():
            sign = torch.sign(difference[upper]); score_difference = (valid_scores[:, None] - valid_scores[None, :])[upper]
            pair_losses.append(functional.softplus(-sign * score_difference).mean())
    pairwise = torch.stack(pair_losses).mean() if pair_losses else scores[mask].sum() * 0.0
    normalized = torch.zeros_like(gains)
    for batch in range(gains.shape[0]):
        values = gains[batch][mask[batch]]; scale = values.std(unbiased=False).clamp_min(1e-6)
        normalized[batch][mask[batch]] = (values - values.mean()) / scale
    regression = functional.smooth_l1_loss(scores[mask], normalized[mask])
    total = listwise + 0.5 * pairwise + 0.1 * regression
    return {"total": total, "listwise": listwise, "pairwise": pairwise, "regression": regression}
