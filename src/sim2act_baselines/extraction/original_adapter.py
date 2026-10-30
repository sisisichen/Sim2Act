from __future__ import annotations

import networkx as nx
import numpy as np
import open3d as o3d
from pc_skeletor import LBC

from sim2act_baselines.upstream.crackstructures_adapter import EXTRACTION_PARAMETERS

from .extraction_result import GraphBuildMetadata
from .graph_builders import build_original_complete_graph_mst, partition_furcations


def contract_points_upstream_equivalent(subcloud: o3d.geometry.PointCloud) -> np.ndarray:
    """Run the unchanged LBC parameters from ENSTRECT.contract_subcloud."""
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


def graph_from_contracted_points_original(
    points: np.ndarray,
) -> tuple[nx.Graph, GraphBuildMetadata]:
    graph, metadata = build_original_complete_graph_mst(points)
    return partition_furcations(graph), metadata
