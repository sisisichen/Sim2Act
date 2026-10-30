from __future__ import annotations

from pathlib import Path
import time

import networkx as nx
import numpy as np
import open3d as o3d
from pc_skeletor import LBC
from sklearn.cluster import DBSCAN

from enstrect.extraction.utils import G_to_obj

from sim2act_baselines.evaluation.extraction_trace import parse_obj_counts
from sim2act_baselines.evaluation.official_tolerance_metrics import graph_length
from sim2act_baselines.upstream.crackstructures_adapter import EXTRACTION_PARAMETERS

from .extraction_result import ExtractionResult
from .graph_builders import build_exact_euclidean_mst, partition_furcations
from .resource_monitor import ResourceMonitor


def _contract_points(subcloud: o3d.geometry.PointCloud) -> np.ndarray:
    # These are the exact parameters in ENSTRECT.contract_subcloud. Only the
    # complete-graph representation after this operation is replaced.
    lbc = LBC(
        point_cloud=subcloud,
        down_sample=0.001,
        init_contraction=EXTRACTION_PARAMETERS["init_contraction"],
        init_attraction=10.0,
        max_attraction=2048,
        step_wise_contraction_amplification="auto",
        termination_ratio=0.003,
        max_iteration_steps=20,
        filter_nb_neighbors=20,
        filter_std_ratio=2.0,
    )
    lbc.extract_skeleton()
    return np.asarray(lbc.contracted_point_cloud.points, dtype=np.float64)


class MemorySafeExtractor:
    implementation = "exact_memory_safe_local"
    equivalence_status = "verified_exact"

    def __init__(self, points: np.ndarray, normals: np.ndarray, colors: np.ndarray):
        points = np.asarray(points, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        colors = np.asarray(colors, dtype=np.float64)
        if points.shape != normals.shape or points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point/normal/color shape mismatch")
        if not (np.isfinite(points).all() and np.isfinite(normals).all() and np.isfinite(colors).all()):
            raise ValueError("non-finite point-cloud values")
        self.points = points
        self.cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        self.cloud.normals = o3d.utility.Vector3dVector(normals)
        self.cloud.colors = o3d.utility.Vector3dVector(colors)

    def extract(
        self,
        fused_label: np.ndarray,
        output_obj: Path,
        *,
        rss_log_path: Path | None = None,
    ) -> ExtractionResult:
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
        cluster_metadata: list[dict] = []
        failures: list[str] = []
        with ResourceMonitor(rss_log_path) as resources:
            if len(positive_indices):
                positive_points = self.points[positive_indices]
                cluster = DBSCAN(
                    eps=EXTRACTION_PARAMETERS["eps_m"],
                    min_samples=EXTRACTION_PARAMETERS["min_samples_cluster"],
                ).fit_predict(positive_points)
                cluster_ids, cluster_counts = np.unique(cluster, return_counts=True)
                keep = cluster_ids != -1
                cluster_ids = cluster_ids[keep]
                cluster_counts = cluster_counts[keep]
                clusters_before = len(cluster_ids)
                valid = cluster_counts >= EXTRACTION_PARAMETERS["min_points"]
                clusters_after = int(valid.sum())
                points_before = int(cluster_counts[valid].sum())
                for cluster_id, cluster_count in zip(cluster_ids[valid], cluster_counts[valid]):
                    source_indices = positive_indices[np.flatnonzero(cluster == cluster_id)]
                    subcloud = self.cloud.select_by_index(source_indices.tolist())
                    try:
                        contracted_points = _contract_points(subcloud)
                        contracted_count += len(contracted_points)
                        tree, metadata = build_exact_euclidean_mst(contracted_points)
                        nodes_before_partition += len(tree.nodes)
                        edges_before_partition += len(tree.edges)
                        tree = partition_furcations(tree)
                        tree = nx.relabel_nodes(
                            tree,
                            {key: index + len(graph) for index, key in enumerate(tree.nodes)},
                            copy=True,
                        )
                        graph = nx.compose(graph, tree)
                        cluster_metadata.append(
                            {
                                "cluster_id": int(cluster_id),
                                "semantic_points": int(cluster_count),
                                "contracted_points": len(contracted_points),
                                **metadata.to_dict(),
                            }
                        )
                    except Exception as error:
                        failures.append(
                            f"cluster={int(cluster_id)} count={int(cluster_count)} "
                            f"{type(error).__name__}: {error}"
                        )
            if len(graph.nodes):
                G_to_obj(graph, output_obj)
            else:
                output_obj.open("x", encoding="utf-8").close()
        vertices, polylines = parse_obj_counts(output_obj)
        return ExtractionResult(
            implementation=self.implementation,
            equivalence_status=self.equivalence_status,
            # ENSTRECT intentionally catches per-cluster contraction failures and
            # keeps valid geometry from the remaining clusters. Preserve that
            # scientific behavior; retain every caught failure in metadata.
            status="MEMORY_SAFE_EXTRACTION_SUCCESS" if len(graph.nodes) else (
                "PIPELINE_ERROR" if failures else "MEMORY_SAFE_EXTRACTION_SUCCESS"
            ),
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
