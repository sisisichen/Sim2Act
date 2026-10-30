from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from .feature_schema import CANDIDATE_FEATURES, GLOBAL_FEATURES
from .oracle_label_generator import OracleLabelSet
from .state_sampler import StateSpec


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DevelopmentDatasetWriter:
    """Two-file transactional writer: deployable features and privileged labels never share a file."""

    def __init__(self, root: Path, split: str, protocol_hash: str, source_manifest_hash: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.feature_final = self.root / f"{split}_features.h5"
        self.label_final = self.root / f"{split}_labels.h5"
        self.feature_partial = Path(str(self.feature_final) + ".partial")
        self.label_partial = Path(str(self.label_final) + ".partial")
        for path in (self.feature_final, self.label_final, self.feature_partial, self.label_partial):
            if path.exists():
                raise FileExistsError(path)
        self.features = h5py.File(self.feature_partial, "x")
        self.labels = h5py.File(self.label_partial, "x")
        for handle, kind in ((self.features, "causal_features_only"), (self.labels, "privileged_oracle_labels_only")):
            handle.attrs["complete"] = False
            handle.attrs["split"] = split
            handle.attrs["kind"] = kind
            handle.attrs["protocol_sha256"] = protocol_hash
            handle.attrs["source_manifest_sha256"] = source_manifest_hash
        self.features.attrs["global_feature_names_json"] = json.dumps(GLOBAL_FEATURES)
        self.features.attrs["candidate_feature_names_json"] = json.dumps(CANDIDATE_FEATURES)
        self.feature_groups = self.features.create_group("states", track_order=True)
        self.label_groups = self.labels.create_group("states", track_order=True)
        self.rows: list[dict] = []
        self._closed = False

    def add(self, spec: StateSpec, candidate_ids: tuple[str, ...], global_features: np.ndarray,
            candidate_features: np.ndarray, labels: OracleLabelSet) -> None:
        if spec.split != self.split or labels.candidate_view_ids != candidate_ids:
            raise ValueError("state split or candidate-label order mismatch")
        if spec.uid in self.feature_groups or spec.uid in self.label_groups:
            raise ValueError(f"duplicate state UID: {spec.uid}")
        global_array = np.asarray(global_features, dtype=np.float32)
        candidate_array = np.asarray(candidate_features, dtype=np.float32)
        if global_array.shape != (len(GLOBAL_FEATURES),) or candidate_array.shape != (len(candidate_ids), len(CANDIDATE_FEATURES)):
            raise ValueError("feature shape mismatch")
        if not np.isfinite(global_array).all() or not np.isfinite(candidate_array).all():
            raise ValueError("non-finite features")
        fg = self.feature_groups.create_group(spec.uid)
        fg.create_dataset("global", data=global_array, compression="gzip", compression_opts=1, fletcher32=True)
        fg.create_dataset("candidates", data=candidate_array, compression="gzip", compression_opts=1, fletcher32=True)
        string = h5py.string_dtype("utf-8")
        fg.create_dataset("candidate_keys", data=np.asarray(candidate_ids, dtype=object), dtype=string)
        lg = self.label_groups.create_group(spec.uid)
        lg.create_dataset("iou_after", data=np.asarray(labels.iou_after, dtype=np.float64), compression="gzip", compression_opts=1, fletcher32=True)
        lg.create_dataset("delta_iou", data=np.asarray(labels.delta_iou, dtype=np.float64), compression="gzip", compression_opts=1, fletcher32=True)
        lg.attrs["iou_current"] = labels.iou_current
        lg.attrs["top_candidate_key"] = labels.top_view_id
        lg.create_dataset("ranking_keys", data=np.asarray(labels.ranking, dtype=object), dtype=string)
        self.rows.append({
            "state_uid": spec.uid,
            "sample_id": spec.sample_id,
            "split": spec.split,
            "trajectory_seed": spec.trajectory_seed,
            "budget": spec.budget,
            "total_views": spec.total_views,
            "candidate_count": len(candidate_ids),
            "selected_set_hash": spec.selected_set_hash,
            "source_policies": ";".join(spec.source_policies),
        })
        self.features.flush(); self.labels.flush()

    def finalize(self) -> dict:
        if self._closed:
            raise RuntimeError("writer is closed")
        sample_counts: dict[str, int] = {}
        for row in self.rows:
            sample_counts[row["sample_id"]] = sample_counts.get(row["sample_id"], 0) + 1
        for row in self.rows:
            row["sample_balancing_weight"] = 1.0 / sample_counts[row["sample_id"]]
        self.features.attrs.modify("complete", True); self.labels.attrs.modify("complete", True)
        self.features.close(); self.labels.close(); self._closed = True
        self.feature_partial.replace(self.feature_final); self.label_partial.replace(self.label_final)
        manifest = self.root / f"{self.split}_state_manifest.csv"
        with manifest.open("x", encoding="utf-8", newline="") as handle:
            columns = ["state_uid", "sample_id", "split", "trajectory_seed", "budget", "total_views", "candidate_count", "selected_set_hash", "source_policies", "sample_balancing_weight"]
            writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(self.rows)
            handle.flush(); os.fsync(handle.fileno())
        payload = {
            "status": "COMPLETE", "split": self.split, "state_count": len(self.rows),
            "sample_count": len(sample_counts), "features": str(self.feature_final), "features_sha256": sha256(self.feature_final),
            "labels": str(self.label_final), "labels_sha256": sha256(self.label_final),
            "manifest": str(manifest), "manifest_sha256": sha256(manifest),
            "sample_balancing": "each sample has total weight 1.0 regardless of state/view count",
        }
        (self.root / f"{self.split}_dataset_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    def abort(self) -> None:
        if not self._closed:
            self.features.close(); self.labels.close(); self._closed = True
