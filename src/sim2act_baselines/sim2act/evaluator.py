from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def evaluate_ranking(scores: np.ndarray, gains: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64); gains = np.asarray(gains, dtype=np.float64)
    if scores.shape != gains.shape or scores.ndim != 1 or not np.isfinite(scores).all() or not np.isfinite(gains).all():
        raise ValueError("invalid ranking arrays")
    predicted = np.argsort(-scores, kind="stable"); oracle = np.argsort(-gains, kind="stable")
    relevance = gains - gains.min()
    k = min(5, len(gains)); discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(relevance[predicted[:k]] * discounts)); ideal = float(np.sum(relevance[oracle[:k]] * discounts))
    pairs = 0; correct = 0
    for i in range(len(gains)):
        for j in range(i + 1, len(gains)):
            if gains[i] == gains[j]: continue
            pairs += 1; correct += int(np.sign(scores[i] - scores[j]) == np.sign(gains[i] - gains[j]))
    correlation = spearmanr(scores, gains).statistic if len(gains) > 1 else 1.0
    if not np.isfinite(correlation): correlation = 0.0
    return {
        "oracle_regret": float(gains.max() - gains[predicted[0]]),
        "ndcg_at_5": float(dcg / ideal) if ideal > 0 else 1.0,
        "oracle_top1_hit": float(predicted[0] == oracle[0]),
        "oracle_top3_recall": float(oracle[0] in predicted[:min(3, len(gains))]),
        "pairwise_accuracy": float(correct / pairs) if pairs else 1.0,
        "spearman": float(correlation),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
