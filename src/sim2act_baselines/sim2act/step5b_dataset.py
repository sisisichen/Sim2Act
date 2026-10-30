from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .access_guard import assert_scaler_source


FILES = {
    "development_train": ("train_state_features.h5", "train_oracle_labels.h5", "train_state_manifest.csv"),
    "development_validation": ("validation_state_features.h5", "validation_oracle_labels.h5", "validation_state_manifest.csv"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Step5BDataset(Dataset):
    """In-memory index over physically separated causal features and privileged labels."""

    def __init__(self, root: Path, split: str, scaler: dict | None = None):
        if split not in FILES:
            raise ValueError(split)
        self.root = Path(root)
        self.split = split
        self.scaler = scaler
        feature_name, label_name, manifest_name = FILES[split]
        self.features_path = self.root / feature_name
        self.labels_path = self.root / label_name
        self.manifest_path = self.root / manifest_name
        with self.manifest_path.open(encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        with h5py.File(self.features_path, "r") as features, h5py.File(self.labels_path, "r") as labels:
            if not bool(features.attrs["complete"]) or not bool(labels.attrs["complete"]):
                raise ValueError("incomplete corpus")
            if features.attrs["protocol_sha256"] != labels.attrs["protocol_sha256"]:
                raise ValueError("feature/label protocol mismatch")
            feature_keys = set(features["states"])
            label_keys = set(labels["states"])
            manifest_keys = {row["state_id"] for row in self.rows}
            if feature_keys != label_keys or feature_keys != manifest_keys:
                raise ValueError("feature/label/manifest alignment failure")
            self.global_names = tuple(json.loads(features.attrs["global_feature_names_json"]))
            self.candidate_names = tuple(json.loads(features.attrs["candidate_feature_names_json"]))
            self.records = []
            for row in self.rows:
                state_id = row["state_id"]
                fg = features["states"][state_id]
                lg = labels["states"][state_id]
                candidate_keys = tuple(_text(value) for value in fg["candidate_ids"][:])
                if len(candidate_keys) != len(lg["delta_iou"]):
                    raise ValueError(f"candidate alignment failure: {state_id}")
                self.records.append({
                    "global": np.asarray(fg["global_features"][:], dtype=np.float32),
                    "candidates": np.asarray(fg["candidate_causal_features"][:], dtype=np.float32),
                    "candidate_keys": candidate_keys,
                    "gains": np.asarray(lg["delta_iou"][:], dtype=np.float32),
                    "iou_after": np.asarray(lg["iou_after"][:], dtype=np.float64),
                    "oracle_best": tuple(_text(value) for value in lg["oracle_best_candidate_set"][:]),
                })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        record = self.records[index]
        global_features = record["global"].copy()
        candidate_features = record["candidates"].copy()
        if self.scaler is not None:
            global_features = _scale(global_features, self.scaler["global_center"], self.scaler["global_scale"], self.scaler)
            candidate_features = _scale(candidate_features, self.scaler["candidate_center"], self.scaler["candidate_scale"], self.scaler)
        return {
            "state_id": row["state_id"],
            "scene_id": row["scene_id"],
            "primary_policy": row["primary_policy"],
            "budget": int(row["state_budget"]),
            "candidate_keys": record["candidate_keys"],
            "oracle_best": record["oracle_best"],
            "global": torch.from_numpy(global_features),
            "candidates": torch.from_numpy(candidate_features),
            "gains": torch.from_numpy(record["gains"].copy()),
            "weight": float(row["scene_balancing_weight"]),
        }


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _scale(values: np.ndarray, center: list[float], scale: list[float], scaler: dict) -> np.ndarray:
    result = (values - np.asarray(center, np.float32)) / np.asarray(scale, np.float32)
    clip = float(scaler.get("clip_standard_deviations", 8.0))
    return np.clip(result, -clip, clip).astype(np.float32, copy=False)


def collate_step5b(batch: list[dict]) -> dict:
    maximum = max(len(row["candidate_keys"]) for row in batch)
    candidate_dim = batch[0]["candidates"].shape[1]
    candidates = torch.zeros(len(batch), maximum, candidate_dim, dtype=torch.float32)
    gains = torch.zeros(len(batch), maximum, dtype=torch.float32)
    mask = torch.zeros(len(batch), maximum, dtype=torch.bool)
    for index, row in enumerate(batch):
        count = len(row["candidate_keys"])
        candidates[index, :count] = row["candidates"]
        gains[index, :count] = row["gains"]
        mask[index, :count] = True
    return {
        "state_id": [row["state_id"] for row in batch],
        "scene_id": [row["scene_id"] for row in batch],
        "candidate_keys": [row["candidate_keys"] for row in batch],
        "oracle_best": [row["oracle_best"] for row in batch],
        "global": torch.stack([row["global"] for row in batch]),
        "candidates": candidates,
        "gains": gains,
        "mask": mask,
        "weight": torch.tensor([row["weight"] for row in batch], dtype=torch.float32),
    }


class ScenePolicyBudgetSampler(Sampler[int]):
    """Uniform scene -> policy -> budget sampling with deterministic epochs."""

    def __init__(self, dataset: Step5BDataset, seed: int, samples_per_epoch: int | None = None):
        if dataset.split != "development_train":
            raise ValueError("scene-balanced sampler is training-only")
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.samples_per_epoch = int(samples_per_epoch or len(dataset))
        nested: dict[str, dict[str, dict[int, list[int]]]] = {}
        for index, row in enumerate(dataset.rows):
            nested.setdefault(row["scene_id"], {}).setdefault(row["primary_policy"], {}).setdefault(int(row["state_budget"]), []).append(index)
        self.nested = nested
        self.scenes = tuple(sorted(nested))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        scene_order: list[str] = []
        while len(scene_order) < self.samples_per_epoch:
            scene_order.extend(list(rng.permutation(self.scenes)))
        for scene_id in scene_order[:self.samples_per_epoch]:
            policies = tuple(sorted(self.nested[scene_id]))
            policy = policies[int(rng.integers(len(policies)))]
            budgets = tuple(sorted(self.nested[scene_id][policy]))
            budget = budgets[int(rng.integers(len(budgets)))]
            choices = self.nested[scene_id][policy][budget]
            yield choices[int(rng.integers(len(choices)))]


def fit_train_only_scaler(dataset: Step5BDataset, feature_schema_sha256: str) -> dict:
    assert_scaler_source(dataset.split)
    globals_ = np.stack([record["global"] for record in dataset.records]).astype(np.float64)
    candidates = np.concatenate([record["candidates"] for record in dataset.records]).astype(np.float64)
    global_scale = np.maximum(globals_.std(axis=0), 1e-6)
    candidate_scale = np.maximum(candidates.std(axis=0), 1e-6)
    payload = {
        "fit_split": "development_train",
        "fitted_state_count": len(dataset),
        "fitted_scene_count": len({row["scene_id"] for row in dataset.rows}),
        "global_center": globals_.mean(axis=0).tolist(),
        "global_scale": global_scale.tolist(),
        "candidate_center": candidates.mean(axis=0).tolist(),
        "candidate_scale": candidate_scale.tolist(),
        "clip_standard_deviations": 8.0,
        "validity_indicator_handling": "frozen neutral value 0.0; no additional indicators in STEP 5A schema",
        "training_manifest_sha256": sha256(dataset.manifest_path),
        "feature_schema_sha256": feature_schema_sha256,
    }
    probe = globals_[: min(64, len(globals_))].astype(np.float32)
    first = _scale(probe, payload["global_center"], payload["global_scale"], payload)
    restored = json.loads(json.dumps(payload, sort_keys=True))
    second = _scale(probe, restored["global_center"], restored["global_scale"], restored)
    payload["roundtrip_max_abs_error"] = float(np.max(np.abs(first - second)))
    payload["roundtrip_reproducible"] = payload["roundtrip_max_abs_error"] == 0.0
    return payload
