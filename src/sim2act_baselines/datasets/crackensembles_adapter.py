from __future__ import annotations

from pathlib import Path

from .crackstructures_adapter import EvaluationScene, InferenceScene


OFFICIAL_SPLIT_RANGES = {
    "train": range(0, 1090),
    "validation": range(1090, 1218),
    "test": range(1218, 1429),
}


def official_split_ids(split: str) -> tuple[str, ...]:
    if split not in OFFICIAL_SPLIT_RANGES:
        raise KeyError(split)
    return tuple(f"{value:04d}" for value in OFFICIAL_SPLIT_RANGES[split])


def official_split_for_id(sample_id: str | int) -> str:
    value = int(sample_id)
    for split, values in OFFICIAL_SPLIT_RANGES.items():
        if value in values:
            return split
    raise ValueError(f"sample ID is outside the published CrackEnsembles range: {sample_id}")


def crackensembles_scene(raw_root: Path, split: str, sample_id: str | int) -> EvaluationScene:
    sample = f"{int(sample_id):04d}"
    official = official_split_for_id(sample)
    if split != official:
        raise ValueError(f"sample {sample} belongs to {official}, not {split}")
    root = raw_root / split / sample
    inference = InferenceScene(
        dataset="crackensembles",
        scene_id=sample,
        scene_root=root,
        mesh_path=root / "mesh/mesh.obj",
        cameras_path=root / "cameras.json",
        views_dir=root / "views",
    )
    scene = EvaluationScene(inference, root / "annotations/crack.obj")
    scene.validate()
    return scene


def deterministic_validation_diagnostic_ids() -> tuple[str, ...]:
    """Eight model-independent validation IDs including first/middle/last.

    Start from eight rounded equal-interval targets across the 128 official
    validation IDs. Replace the target nearest the lower middle with that exact
    lower-middle index; ties are resolved toward the smaller target.
    """

    ids = official_split_ids("validation")
    targets = [0, 18, 36, 54, 73, 91, 109, 127]
    middle = (len(ids) - 1) // 2
    replace = min(range(len(targets)), key=lambda index: (abs(targets[index] - middle), index))
    targets[replace] = middle
    result = tuple(ids[index] for index in sorted(set(targets)))
    if len(result) != 8 or result[0] != ids[0] or result[-1] != ids[-1] or ids[middle] not in result:
        raise RuntimeError(result)
    return result


def deterministic_crackensembles_subset(raw_root: Path, maximum: int = 3) -> list[EvaluationScene]:
    """Return the first lexicographically sorted test scenes, independent of GT/results."""

    if not 1 <= maximum <= 3:
        raise ValueError("STEP 3A CrackEnsembles subset must contain 1 to 3 scenes")
    test_root = raw_root / "test"
    roots = sorted(path for path in test_root.iterdir() if path.is_dir())[:maximum]
    scenes: list[EvaluationScene] = []
    for root in roots:
        scenes.append(crackensembles_scene(raw_root, "test", root.name))
    if len(scenes) != maximum:
        raise RuntimeError(f"expected {maximum} deterministic scenes, found {len(scenes)}")
    return scenes
