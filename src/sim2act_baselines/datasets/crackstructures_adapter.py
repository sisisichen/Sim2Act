from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceScene:
    """Inputs allowed to probability/geometry caching; deliberately no GT field."""

    dataset: str
    scene_id: str
    scene_root: Path
    mesh_path: Path
    cameras_path: Path
    views_dir: Path

    def validate(self) -> None:
        required = (self.scene_root, self.mesh_path, self.cameras_path, self.views_dir)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"inference scene is incomplete: {missing}")
        if "annotations" in {part.lower() for path in required for part in path.parts}:
            raise ValueError("ground-truth path leaked into InferenceScene")


@dataclass(frozen=True)
class EvaluationScene:
    inference: InferenceScene
    annotation_path: Path

    def validate(self) -> None:
        self.inference.validate()
        if not self.annotation_path.is_file():
            raise FileNotFoundError(self.annotation_path)


def crackstructures_indoors_segment3(raw_root: Path) -> EvaluationScene:
    root = raw_root / "indoors/segment3"
    inference = InferenceScene(
        dataset="crackstructures",
        scene_id="indoors_segment3",
        scene_root=root,
        mesh_path=root / "mesh/mesh.obj",
        cameras_path=root / "cameras.json",
        views_dir=root / "views",
    )
    scene = EvaluationScene(inference, root / "annotations/crack.obj")
    scene.validate()
    return scene

