from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetinaConfig:
    data_flag: str = "retinamnist"
    data_root: Path = Path("data")
    export_dir: Path = Path("data/retinamnist")
    classes: tuple[int, ...] = (0, 1, 2, 3, 4)
    seed: int = 42


@dataclass(frozen=True)
class DDPMConfig:
    train_script: Path = Path("diffusers/examples/unconditional_image_generation/train_unconditional.py")
    output_root: Path = Path("outputs/ddpm")
    resolution: int = 32
    train_batch_size: int = 64
    num_epochs: int = 100
    mixed_precision: str = "fp16"
    logger: str | None = "wandb"


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 64
    max_batches: int = 10
    num_inference_steps: int = 50
    use_fp16: bool = True

