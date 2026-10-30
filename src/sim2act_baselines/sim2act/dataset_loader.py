from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ViewRankingDataset(Dataset):
    def __init__(self, root: Path, split: str, scaler: dict | None = None):
        self.root = Path(root); self.split = split; self.scaler = scaler
        self.features_path = self.root / f"{split}_features.h5"
        self.labels_path = self.root / f"{split}_labels.h5"
        with (self.root / f"{split}_state_manifest.csv").open(encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        with h5py.File(self.features_path, "r") as handle:
            if not bool(handle.attrs["complete"]): raise ValueError("incomplete feature file")
            self.global_names = tuple(json.loads(handle.attrs["global_feature_names_json"]))
            self.candidate_names = tuple(json.loads(handle.attrs["candidate_feature_names_json"]))
        with h5py.File(self.labels_path, "r") as handle:
            if not bool(handle.attrs["complete"]): raise ValueError("incomplete label file")
        self.records = []
        with h5py.File(self.features_path, "r") as features, h5py.File(self.labels_path, "r") as labels:
            for row in self.rows:
                uid = row["state_uid"]; fg = features["states"][uid]; lg = labels["states"][uid]
                self.records.append({
                    "global": np.asarray(fg["global"][:], dtype=np.float32),
                    "candidates": np.asarray(fg["candidates"][:], dtype=np.float32),
                    "keys": tuple(value.decode() if isinstance(value, bytes) else str(value) for value in fg["candidate_keys"][:]),
                    "gains": np.asarray(lg["delta_iou"][:], dtype=np.float32),
                })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]; uid = row["state_uid"]
        record = self.records[index]
        global_features = record["global"].copy(); candidates = record["candidates"].copy()
        keys = record["keys"]; gains = record["gains"]
        if self.scaler is not None:
            global_features = (global_features - np.asarray(self.scaler["global_mean"], np.float32)) / np.asarray(self.scaler["global_scale"], np.float32)
            candidates = (candidates - np.asarray(self.scaler["candidate_mean"], np.float32)) / np.asarray(self.scaler["candidate_scale"], np.float32)
        return {
            "uid": uid, "sample_id": row["sample_id"], "candidate_keys": keys,
            "global": torch.from_numpy(global_features), "candidates": torch.from_numpy(candidates),
            "gains": torch.from_numpy(gains), "weight": float(row["sample_balancing_weight"]),
        }


def collate_variable_candidates(batch: list[dict]) -> dict:
    size = len(batch); maximum = max(len(row["candidate_keys"]) for row in batch)
    global_features = torch.stack([row["global"] for row in batch])
    candidate_features = torch.zeros(size, maximum, batch[0]["candidates"].shape[1], dtype=torch.float32)
    gains = torch.zeros(size, maximum, dtype=torch.float32); mask = torch.zeros(size, maximum, dtype=torch.bool)
    keys = []
    for index, row in enumerate(batch):
        count = len(row["candidate_keys"]); candidate_features[index, :count] = row["candidates"]
        gains[index, :count] = row["gains"]; mask[index, :count] = True; keys.append(row["candidate_keys"])
    return {
        "uid": [row["uid"] for row in batch], "sample_id": [row["sample_id"] for row in batch], "candidate_keys": keys,
        "global": global_features, "candidates": candidate_features, "gains": gains, "mask": mask,
        "weight": torch.tensor([row["weight"] for row in batch], dtype=torch.float32),
    }


def fit_feature_scaler(dataset: ViewRankingDataset, train_manifest_sha256: str) -> dict:
    if dataset.split != "development_train":
        raise ValueError("feature scaler may only be fitted on development_train")
    globals_, candidates_ = [], []
    for index in range(len(dataset)):
        row = dataset[index]; globals_.append(row["global"].numpy()); candidates_.append(row["candidates"].numpy())
    global_array = np.stack(globals_).astype(np.float64); candidate_array = np.concatenate(candidates_).astype(np.float64)
    gscale = np.maximum(global_array.std(axis=0), 1e-6); cscale = np.maximum(candidate_array.std(axis=0), 1e-6)
    return {
        "fit_split": dataset.split, "fit_state_count": len(dataset), "fit_sample_count": len(set(row["sample_id"] for row in dataset.rows)),
        "train_manifest_sha256": train_manifest_sha256,
        "global_mean": global_array.mean(axis=0).tolist(), "global_scale": gscale.tolist(),
        "candidate_mean": candidate_array.mean(axis=0).tolist(), "candidate_scale": cscale.tolist(),
    }
