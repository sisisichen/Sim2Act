from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GraphBuildMetadata:
    implementation: str
    equivalence_status: str
    algorithm: str
    input_nodes: int
    unique_nodes: int
    candidate_edges: int
    mst_edges: int
    affine_rank: int
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractionResult:
    implementation: str
    equivalence_status: str
    status: str
    semantic_positive_count: int
    clusters_before_filter: int
    clusters_after_filter: int
    points_before_contraction: int
    contracted_points: int
    graph_nodes_before_partition: int
    graph_edges_before_partition: int
    final_vertices: int
    final_polylines: int
    predicted_centerline_length_m: float
    runtime_seconds: float
    peak_rss_bytes: int
    cluster_metadata: tuple[dict[str, Any], ...]
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
