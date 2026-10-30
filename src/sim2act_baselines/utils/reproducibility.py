from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 2025) -> dict[str, object]:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "external_seed": seed,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_deterministic_requested": True,
        "deterministic_algorithms_warn_only": True,
        "known_upstream_mesh_seed": 42,
    }

