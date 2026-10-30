from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from sim2act_baselines.active.base import stable_view_key

from .feature_extractor import CausalFeatureExtractor
from .state_sampler import CausalState


@dataclass(frozen=True)
class Sim2ActDecision:
    view_id: str
    score: float
    runtime_seconds: float


class Sim2ActClosedLoopSelector:
    name = "Sim2ActViewRanker"

    def __init__(self, model, scaler: dict, extractor: CausalFeatureExtractor, device: torch.device):
        self.model = model.to(device).eval(); self.scaler = scaler; self.extractor = extractor; self.device = device

    def select(self, state: CausalState) -> Sim2ActDecision:
        started = time.perf_counter(); keys, global_features, candidate_features = self.extractor.extract(state)
        global_features = (global_features - np.asarray(self.scaler["global_mean"], np.float32)) / np.asarray(self.scaler["global_scale"], np.float32)
        candidate_features = (candidate_features - np.asarray(self.scaler["candidate_mean"], np.float32)) / np.asarray(self.scaler["candidate_scale"], np.float32)
        with torch.no_grad():
            global_tensor = torch.from_numpy(global_features).unsqueeze(0).to(self.device)
            candidate_tensor = torch.from_numpy(candidate_features).unsqueeze(0).to(self.device)
            mask = torch.ones(1, len(keys), dtype=torch.bool, device=self.device)
            scores = self.model(global_tensor, candidate_tensor, mask)[0].cpu().numpy()
        best = min(range(len(keys)), key=lambda index: (-float(scores[index]), stable_view_key(keys[index])))
        return Sim2ActDecision(keys[best], float(scores[best]), time.perf_counter() - started)
