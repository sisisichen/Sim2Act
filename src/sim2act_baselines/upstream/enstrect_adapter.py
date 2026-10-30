from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from pytorch3d.io import IO

from enstrect.datasets.multiview import MultiviewDataset
from enstrect.datasets.utils import sample_points_from_meshes_pyt3d
from enstrect.extraction.contraction import extract_centerlines
from enstrect.extraction.utils import G_to_obj, pynt_to_pyt3d, pyt3d_to_pynt
from enstrect.mapping.mapper import Mapper

from .crackstructures_adapter import EXTRACTION_PARAMETERS


def sample_official_geometry(mesh_path: Path, num_points: int = 1_000_000):
    mesh = IO().load_mesh(mesh_path)
    sampled = sample_points_from_meshes_pyt3d(mesh, num_points=num_points)
    return sampled, pyt3d_to_pynt(sampled)


def make_multiview(cameras_path: Path, views_dir: Path, scale: float):
    return MultiviewDataset(cameras_path, views_dir, select_views=None, scale=scale)


def mapper_pointcloud(pcd_pynt: PyntCloud, device: str = "cuda:0"):
    return pynt_to_pyt3d(pcd_pynt, device=device)


def point_frame(points: np.ndarray, normals: np.ndarray, colors: np.ndarray) -> PyntCloud:
    frame = pd.DataFrame(
        np.concatenate((points, normals, colors), axis=1),
        columns=["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"],
    )
    return PyntCloud(frame)


def extract_crack_obj(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    fused_probability: np.ndarray,
    fused_label: np.ndarray,
    output_path: Path,
):
    cloud = point_frame(points, normals, colors)
    cloud.points["argmax"] = fused_label.astype(np.uint8)
    cloud.points["background"] = (fused_label == 0).astype(np.uint8)
    cloud.points["crack"] = (fused_label == 1).astype(np.uint8)
    cloud.points["background_probabilities"] = fused_probability[:, 0].astype(np.float32)
    cloud.points["crack_probabilities"] = fused_probability[:, 1].astype(np.float32)
    graph = extract_centerlines(cloud, category="crack", **EXTRACTION_PARAMETERS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    G_to_obj(graph, output_path)
    if not output_path.exists():
        output_path.write_text("", encoding="utf-8")
    return graph


__all__ = [
    "Mapper",
    "extract_crack_obj",
    "make_multiview",
    "mapper_pointcloud",
    "point_frame",
    "sample_official_geometry",
]

