from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import networkx as nx
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

from enstrect.extraction.contraction import contract_subcloud
from enstrect.extraction.utils import G_to_obj

from sim2act_baselines.upstream.crackstructures_adapter import EXTRACTION_PARAMETERS


@dataclass(frozen=True)
class ExtractionTrace:
    clusters_before_filter: int
    clusters_after_filter: int
    points_before_contraction: int
    points_after_contraction: int
    final_vertices: int
    final_polylines: int
    contraction_failure_count: int
    contraction_failures: tuple[str, ...]
    runtime_seconds: float

    def to_dict(self):
        return asdict(self)


def parse_obj_counts(path: Path) -> tuple[int, int]:
    vertices = 0
    polylines = 0
    if not path.exists():
        return vertices, polylines
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split()
        if fields and fields[0] == "v" and len(fields) >= 4:
            vertices += 1
        elif fields and fields[0] == "l" and len(fields) >= 3:
            polylines += 1
    return vertices, polylines


class ExtractionTracer:
    def __init__(self, points: np.ndarray, normals: np.ndarray, colors: np.ndarray):
        points = np.asarray(points, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        colors = np.asarray(colors, dtype=np.float64)
        if points.shape != normals.shape or points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point/normal/color shape mismatch")
        self.cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        self.cloud.normals = o3d.utility.Vector3dVector(normals)
        self.cloud.colors = o3d.utility.Vector3dVector(colors)
        self.points = points

    def trace(self, fused_label: np.ndarray, output_obj: Path) -> ExtractionTrace:
        if output_obj.exists():
            raise FileExistsError(output_obj)
        output_obj.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        labels = np.asarray(fused_label).reshape(-1)
        if len(labels) != len(self.points):
            raise ValueError("label/geometry alignment mismatch")
        positive_indices = np.flatnonzero(labels == 1)
        graph = nx.Graph()
        clusters_before = 0
        clusters_after = 0
        points_before = 0
        failures: list[str] = []
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
                    contracted = contract_subcloud(
                        subcloud,
                        init_contraction=EXTRACTION_PARAMETERS["init_contraction"],
                    )
                    contracted = nx.relabel_nodes(
                        contracted,
                        {key: index + len(graph) for index, key in enumerate(contracted.nodes)},
                        copy=True,
                    )
                    graph = nx.compose(graph, contracted)
                except Exception as error:
                    failures.append(
                        f"cluster={int(cluster_id)} count={int(cluster_count)} {type(error).__name__}: {error}"
                    )
        if len(graph.nodes):
            G_to_obj(graph, output_obj)
        else:
            output_obj.open("x", encoding="utf-8").close()
        vertices, polylines = parse_obj_counts(output_obj)
        return ExtractionTrace(
            clusters_before_filter=int(clusters_before),
            clusters_after_filter=int(clusters_after),
            points_before_contraction=int(points_before),
            points_after_contraction=int(len(graph.nodes)),
            final_vertices=int(vertices),
            final_polylines=int(polylines),
            contraction_failure_count=len(failures),
            contraction_failures=tuple(failures),
            runtime_seconds=time.perf_counter() - started,
        )
