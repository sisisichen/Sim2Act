from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Iterator

import networkx as nx
import numpy as np
import open3d as o3d
from pc_skeletor import LBC
import robust_laplacian
from sklearn.cluster import DBSCAN

from enstrect.extraction.utils import G_to_obj

from sim2act_baselines.evaluation.extraction_trace import parse_obj_counts
from sim2act_baselines.evaluation.official_tolerance_metrics import graph_length
from sim2act_baselines.extraction.extraction_result import ExtractionResult
from sim2act_baselines.extraction.graph_builders import (
    build_exact_euclidean_mst,
    partition_furcations,
)
from sim2act_baselines.extraction.resource_monitor import ResourceMonitor
from sim2act_baselines.upstream.crackstructures_adapter import EXTRACTION_PARAMETERS


PROTOCOL_VERSION = "geometry_small_cluster_v1"
FIXED_NEIGHBOUR_COUNT = 30
MINIMUM_FIXED_K_INPUT_COUNT = FIXED_NEIGHBOUR_COUNT + 1
VOXEL_SIZE_M = 0.001
FILTER_NB_NEIGHBORS = 20
FILTER_STD_RATIO = 2.0
FALLBACK_TYPES = (
    "NONE",
    "RAW_CLUSTER_FIXED_K",
    "RAW_CLUSTER_ADAPTIVE_K",
    "VALID_EMPTY_TOO_FEW_POINTS",
)
FALLBACK_PRECEDENCE = {
    "NONE": 0,
    "RAW_CLUSTER_FIXED_K": 1,
    "RAW_CLUSTER_ADAPTIVE_K": 2,
    "VALID_EMPTY_TOO_FEW_POINTS": 3,
}


def _lbc(
    cloud: o3d.geometry.PointCloud,
    *,
    down_sample: float,
    filter_nb_neighbors: int,
    filter_std_ratio: float,
) -> LBC:
    return LBC(
        point_cloud=cloud,
        down_sample=down_sample,
        init_contraction=EXTRACTION_PARAMETERS["init_contraction"],
        init_attraction=10.0,
        max_attraction=2048,
        step_wise_contraction_amplification="auto",
        termination_ratio=0.003,
        max_iteration_steps=20,
        filter_nb_neighbors=filter_nb_neighbors,
        filter_std_ratio=filter_std_ratio,
    )


def _coordinate_rank(points: np.ndarray) -> tuple[int, int]:
    points = np.asarray(points, dtype=np.float64)
    unique = np.unique(points, axis=0)
    if len(unique) < 2:
        return len(unique), 0
    centered = unique - unique.mean(axis=0, keepdims=True)
    return len(unique), int(np.linalg.matrix_rank(centered))


@dataclass(frozen=True)
class PreparedCluster:
    raw_cluster_point_count: int
    raw_unique_point_count: int
    centered_coordinate_rank: int
    downsampled_point_count: int
    effective_neighbour_count: int
    fallback_type: str
    lbc: LBC | None

    @property
    def small_cluster_fallback_used(self) -> bool:
        return self.fallback_type != "NONE"

    @property
    def valid_empty(self) -> bool:
        return self.fallback_type == "VALID_EMPTY_TOO_FEW_POINTS"

    def metadata(self) -> dict[str, Any]:
        return {
            "geometry_protocol_version": PROTOCOL_VERSION,
            "small_cluster_fallback_used": self.small_cluster_fallback_used,
            "fallback_type": self.fallback_type,
            "raw_cluster_point_count": self.raw_cluster_point_count,
            "raw_unique_point_count": self.raw_unique_point_count,
            "centered_coordinate_rank": self.centered_coordinate_rank,
            "downsampled_point_count": self.downsampled_point_count,
            "effective_neighbour_count": self.effective_neighbour_count,
        }


def prepare_cluster(subcloud: o3d.geometry.PointCloud) -> PreparedCluster:
    raw_points = np.asarray(subcloud.points, dtype=np.float64)
    if raw_points.ndim != 2 or raw_points.shape[1:] != (3,):
        raise ValueError("raw cluster points must have shape (N, 3)")
    if not np.isfinite(raw_points).all():
        raise ValueError("raw cluster contains non-finite coordinates")
    raw_count = len(raw_points)
    unique_count, rank = _coordinate_rank(raw_points)

    # Reproduce the original LBC constructor preprocessing exactly to obtain
    # the actual input count for pc_skeletor's first fixed-k Laplacian.
    normal_lbc = _lbc(
        subcloud,
        down_sample=VOXEL_SIZE_M,
        filter_nb_neighbors=FILTER_NB_NEIGHBORS,
        filter_std_ratio=FILTER_STD_RATIO,
    )
    downsampled_count = len(normal_lbc.pcd.points)
    if downsampled_count >= MINIMUM_FIXED_K_INPUT_COUNT:
        return PreparedCluster(
            raw_cluster_point_count=raw_count,
            raw_unique_point_count=unique_count,
            centered_coordinate_rank=rank,
            downsampled_point_count=downsampled_count,
            effective_neighbour_count=FIXED_NEIGHBOUR_COUNT,
            fallback_type="NONE",
            lbc=normal_lbc,
        )

    if raw_count < 3 or unique_count < 3 or rank < 2:
        return PreparedCluster(
            raw_cluster_point_count=raw_count,
            raw_unique_point_count=unique_count,
            centered_coordinate_rank=rank,
            downsampled_point_count=downsampled_count,
            effective_neighbour_count=max(0, min(FIXED_NEIGHBOUR_COUNT, raw_count - 1)),
            fallback_type="VALID_EMPTY_TOO_FEW_POINTS",
            lbc=None,
        )

    # "Raw cluster" is deliberately the unvoxelized and unfiltered DBSCAN
    # cluster. Only this preprocessing bypass and, for N < 31, k are amended.
    raw_lbc = _lbc(
        subcloud,
        down_sample=-1,
        filter_nb_neighbors=0,
        filter_std_ratio=0.0,
    )
    if raw_count >= MINIMUM_FIXED_K_INPUT_COUNT:
        fallback_type = "RAW_CLUSTER_FIXED_K"
        effective_k = FIXED_NEIGHBOUR_COUNT
    else:
        fallback_type = "RAW_CLUSTER_ADAPTIVE_K"
        effective_k = min(FIXED_NEIGHBOUR_COUNT, raw_count - 1)
    return PreparedCluster(
        raw_cluster_point_count=raw_count,
        raw_unique_point_count=unique_count,
        centered_coordinate_rank=rank,
        downsampled_point_count=downsampled_count,
        effective_neighbour_count=effective_k,
        fallback_type=fallback_type,
        lbc=raw_lbc,
    )


@contextmanager
def _effective_neighbour_override(effective_k: int) -> Iterator[None]:
    original = robust_laplacian.point_cloud_laplacian

    def amended_point_cloud_laplacian(
        points: np.ndarray,
        mollify_factor: float = 1e-5,
        n_neighbors: int = FIXED_NEIGHBOUR_COUNT,
    ):
        del n_neighbors
        return original(
            points,
            mollify_factor=mollify_factor,
            n_neighbors=effective_k,
        )

    robust_laplacian.point_cloud_laplacian = amended_point_cloud_laplacian
    try:
        yield
    finally:
        robust_laplacian.point_cloud_laplacian = original


def contract_prepared_cluster(prepared: PreparedCluster) -> np.ndarray:
    if prepared.valid_empty:
        return np.empty((0, 3), dtype=np.float64)
    if prepared.lbc is None:
        raise RuntimeError("non-empty prepared cluster is missing LBC")
    if prepared.fallback_type == "RAW_CLUSTER_ADAPTIVE_K":
        with _effective_neighbour_override(prepared.effective_neighbour_count):
            prepared.lbc.extract_skeleton()
    else:
        prepared.lbc.extract_skeleton()
    points = np.asarray(
        prepared.lbc.contracted_point_cloud.points, dtype=np.float64
    )
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise RuntimeError("contracted points have invalid shape")
    if not np.isfinite(points).all():
        raise RuntimeError("contracted points contain NaN or Inf")
    return points


@dataclass(frozen=True)
class AmendedExtractionResult:
    extraction: ExtractionResult
    geometry_protocol_version: str
    small_cluster_fallback_used: bool
    fallback_type: str
    raw_cluster_point_count: int | None
    downsampled_point_count: int | None
    effective_neighbour_count: int | None
    cluster_protocol_metadata: tuple[dict[str, Any], ...]
    fallback_counts_by_type: dict[str, int]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.extraction, name)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.extraction.to_dict(),
            "geometry_protocol_version": self.geometry_protocol_version,
            "small_cluster_fallback_used": self.small_cluster_fallback_used,
            "fallback_type": self.fallback_type,
            "raw_cluster_point_count": self.raw_cluster_point_count,
            "downsampled_point_count": self.downsampled_point_count,
            "effective_neighbour_count": self.effective_neighbour_count,
            "cluster_protocol_metadata": list(self.cluster_protocol_metadata),
            "fallback_counts_by_type": self.fallback_counts_by_type,
        }


def _summary(
    metadata: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None, dict[str, int]]:
    counts = {name: 0 for name in FALLBACK_TYPES}
    for row in metadata:
        counts[row["fallback_type"]] += 1
    if not metadata:
        return "NONE", None, counts
    selected_type = max(
        (row["fallback_type"] for row in metadata),
        key=lambda value: FALLBACK_PRECEDENCE[value],
    )
    representative = next(
        row for row in metadata if row["fallback_type"] == selected_type
    )
    return selected_type, representative, counts


class AmendedMemorySafeExtractor:
    implementation = "exact_memory_safe_local_geometry_small_cluster_v1"
    equivalence_status = "verified_exact_normal_path"

    def __init__(self, points: np.ndarray, normals: np.ndarray, colors: np.ndarray):
        points = np.asarray(points, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        colors = np.asarray(colors, dtype=np.float64)
        if (
            points.shape != normals.shape
            or points.shape != colors.shape
            or points.ndim != 2
            or points.shape[1] != 3
        ):
            raise ValueError("point/normal/color shape mismatch")
        if not (
            np.isfinite(points).all()
            and np.isfinite(normals).all()
            and np.isfinite(colors).all()
        ):
            raise ValueError("non-finite point-cloud values")
        self.points = points
        self.cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        self.cloud.normals = o3d.utility.Vector3dVector(normals)
        self.cloud.colors = o3d.utility.Vector3dVector(colors)

    def inspect_cluster_protocol(
        self,
        fused_label: np.ndarray,
    ) -> tuple[dict[str, Any], ...]:
        labels = np.asarray(fused_label).reshape(-1)
        if len(labels) != len(self.points):
            raise ValueError("label/geometry alignment mismatch")
        positive_indices = np.flatnonzero(labels == 1)
        if not len(positive_indices):
            return ()
        cluster = DBSCAN(
            eps=EXTRACTION_PARAMETERS["eps_m"],
            min_samples=EXTRACTION_PARAMETERS["min_samples_cluster"],
        ).fit_predict(self.points[positive_indices])
        cluster_ids, cluster_counts = np.unique(cluster, return_counts=True)
        rows: list[dict[str, Any]] = []
        for cluster_id, cluster_count in zip(cluster_ids, cluster_counts):
            if (
                cluster_id == -1
                or cluster_count < EXTRACTION_PARAMETERS["min_points"]
            ):
                continue
            source_indices = positive_indices[
                np.flatnonzero(cluster == cluster_id)
            ]
            subcloud = self.cloud.select_by_index(source_indices.tolist())
            prepared = prepare_cluster(subcloud)
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    **prepared.metadata(),
                }
            )
        return tuple(rows)

    def extract(
        self,
        fused_label: np.ndarray,
        output_obj: Path,
        *,
        rss_log_path: Path | None = None,
    ) -> AmendedExtractionResult:
        output_obj = Path(output_obj)
        if output_obj.exists():
            raise FileExistsError(output_obj)
        output_obj.parent.mkdir(parents=True, exist_ok=True)
        labels = np.asarray(fused_label).reshape(-1)
        if len(labels) != len(self.points):
            raise ValueError("label/geometry alignment mismatch")
        positive_indices = np.flatnonzero(labels == 1)
        started = time.perf_counter()
        graph = nx.Graph()
        clusters_before = clusters_after = points_before = contracted_count = 0
        nodes_before_partition = edges_before_partition = 0
        cluster_metadata: list[dict[str, Any]] = []
        protocol_metadata: list[dict[str, Any]] = []
        failures: list[str] = []
        valid_empty_clusters = 0
        with ResourceMonitor(rss_log_path) as resources:
            if len(positive_indices):
                positive_points = self.points[positive_indices]
                cluster = DBSCAN(
                    eps=EXTRACTION_PARAMETERS["eps_m"],
                    min_samples=EXTRACTION_PARAMETERS["min_samples_cluster"],
                ).fit_predict(positive_points)
                cluster_ids, cluster_counts = np.unique(
                    cluster, return_counts=True
                )
                keep = cluster_ids != -1
                cluster_ids = cluster_ids[keep]
                cluster_counts = cluster_counts[keep]
                clusters_before = len(cluster_ids)
                valid = cluster_counts >= EXTRACTION_PARAMETERS["min_points"]
                clusters_after = int(valid.sum())
                points_before = int(cluster_counts[valid].sum())
                for cluster_id, cluster_count in zip(
                    cluster_ids[valid], cluster_counts[valid]
                ):
                    source_indices = positive_indices[
                        np.flatnonzero(cluster == cluster_id)
                    ]
                    subcloud = self.cloud.select_by_index(source_indices.tolist())
                    prepared: PreparedCluster | None = None
                    try:
                        prepared = prepare_cluster(subcloud)
                        protocol_row = {
                            "cluster_id": int(cluster_id),
                            **prepared.metadata(),
                        }
                        protocol_metadata.append(protocol_row)
                        contracted_points = contract_prepared_cluster(prepared)
                        if prepared.valid_empty:
                            valid_empty_clusters += 1
                            cluster_metadata.append(
                                {
                                    "cluster_id": int(cluster_id),
                                    "semantic_points": int(cluster_count),
                                    "contracted_points": 0,
                                    "status": "VALID_EMPTY_TOO_FEW_POINTS",
                                    **prepared.metadata(),
                                }
                            )
                            continue
                        contracted_count += len(contracted_points)
                        tree, metadata = build_exact_euclidean_mst(
                            contracted_points
                        )
                        nodes_before_partition += len(tree.nodes)
                        edges_before_partition += len(tree.edges)
                        tree = partition_furcations(tree)
                        tree = nx.relabel_nodes(
                            tree,
                            {
                                key: index + len(graph)
                                for index, key in enumerate(tree.nodes)
                            },
                            copy=True,
                        )
                        graph = nx.compose(graph, tree)
                        cluster_metadata.append(
                            {
                                "cluster_id": int(cluster_id),
                                "semantic_points": int(cluster_count),
                                "contracted_points": len(contracted_points),
                                **metadata.to_dict(),
                                **prepared.metadata(),
                            }
                        )
                    except Exception as error:
                        detail = (
                            f"cluster={int(cluster_id)} count={int(cluster_count)} "
                            f"{type(error).__name__}: {error}"
                        )
                        if prepared is not None:
                            detail += (
                                f" fallback={prepared.fallback_type}"
                                f" downsampled={prepared.downsampled_point_count}"
                                f" effective_k={prepared.effective_neighbour_count}"
                            )
                        failures.append(detail)
            if len(graph.nodes):
                G_to_obj(graph, output_obj)
            else:
                output_obj.open("x", encoding="utf-8").close()
        vertices, polylines = parse_obj_counts(output_obj)
        if len(graph.nodes):
            status = "MEMORY_SAFE_EXTRACTION_SUCCESS"
        elif failures:
            status = "PIPELINE_ERROR"
        elif valid_empty_clusters:
            status = "VALID_EMPTY_TOO_FEW_POINTS"
        else:
            status = "MEMORY_SAFE_EXTRACTION_SUCCESS"
        base = ExtractionResult(
            implementation=self.implementation,
            equivalence_status=self.equivalence_status,
            status=status,
            semantic_positive_count=len(positive_indices),
            clusters_before_filter=int(clusters_before),
            clusters_after_filter=int(clusters_after),
            points_before_contraction=int(points_before),
            contracted_points=int(contracted_count),
            graph_nodes_before_partition=int(nodes_before_partition),
            graph_edges_before_partition=int(edges_before_partition),
            final_vertices=int(vertices),
            final_polylines=int(polylines),
            predicted_centerline_length_m=float(graph_length(graph)),
            runtime_seconds=time.perf_counter() - started,
            peak_rss_bytes=resources.peak_rss_bytes,
            cluster_metadata=tuple(cluster_metadata),
            failures=tuple(failures),
        )
        fallback_type, representative, fallback_counts = _summary(
            protocol_metadata
        )
        return AmendedExtractionResult(
            extraction=base,
            geometry_protocol_version=PROTOCOL_VERSION,
            small_cluster_fallback_used=fallback_type != "NONE",
            fallback_type=fallback_type,
            raw_cluster_point_count=(
                None
                if representative is None
                else representative["raw_cluster_point_count"]
            ),
            downsampled_point_count=(
                None
                if representative is None
                else representative["downsampled_point_count"]
            ),
            effective_neighbour_count=(
                None
                if representative is None
                else representative["effective_neighbour_count"]
            ),
            cluster_protocol_metadata=tuple(protocol_metadata),
            fallback_counts_by_type=fallback_counts,
        )
