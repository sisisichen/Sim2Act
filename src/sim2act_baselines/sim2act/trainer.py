from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset_loader import collate_variable_candidates
from .evaluator import evaluate_ranking, mean_metrics
from .losses import ranking_loss


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def evaluate_model(model, dataset, device: torch.device) -> dict[str, float]:
    model.eval(); rows = []
    for item in DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_variable_candidates):
        scores = model(item["global"].to(device), item["candidates"].to(device), item["mask"].to(device)).cpu().numpy()
        for index, count in enumerate(item["mask"].sum(dim=1).tolist()):
            rows.append(evaluate_ranking(scores[index, :count], item["gains"][index, :count].numpy()))
    return mean_metrics(rows)


def train_smoke(model, train_dataset, validation_dataset, output: Path, seed: int, device: torch.device,
                max_epochs: int = 100, patience: int = 12) -> dict:
    seed_everything(seed); model.to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=8, shuffle=True, generator=generator, collate_fn=collate_variable_candidates)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = float("inf"); best_state = None; best_metrics = None; epochs_without = 0; history = []
    for epoch in range(1, max_epochs + 1):
        model.train(); losses = []
        for item in loader:
            optimizer.zero_grad(set_to_none=True)
            scores = model(item["global"].to(device), item["candidates"].to(device), item["mask"].to(device))
            values = ranking_loss(scores, item["gains"].to(device), item["mask"].to(device))
            if not torch.isfinite(values["total"]): raise RuntimeError("non-finite training loss")
            values["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(values["total"].detach().cpu()))
        metrics = evaluate_model(model, validation_dataset, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        if metrics["oracle_regret"] < best - 1e-12:
            best = metrics["oracle_regret"]; best_state = copy.deepcopy(model.state_dict()); best_metrics = metrics; epochs_without = 0
        else:
            epochs_without += 1
        if epochs_without >= patience: break
    if best_state is None: raise RuntimeError("no valid checkpoint")
    model.load_state_dict(best_state); output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": best_state, "seed": seed, "global_dim": model.global_dim, "candidate_dim": model.candidate_dim,
        "parameter_count": model.trainable_parameter_count, "best_metrics": best_metrics, "history": history,
    }
    torch.save(payload, output)
    summary = {"seed": seed, "checkpoint": str(output), "epochs": len(history), "best_metrics": best_metrics, "parameter_count": model.trainable_parameter_count}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
