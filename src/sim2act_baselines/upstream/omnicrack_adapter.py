from __future__ import annotations

from pathlib import Path

from omnicrack30k.inference import OmniCrack30kModel


def build_official_model(allow_tqdm: bool = True) -> OmniCrack30kModel:
    return OmniCrack30kModel(folds=(0, 1, 2, 4), allow_tqdm=allow_tqdm)


def checkpoint_zip() -> Path:
    import omnicrack30k.inference as inference_module

    return Path(inference_module.__file__).parent / "checkpoints/nnUNetTrainer__nnUNetPlans__2d.zip"
