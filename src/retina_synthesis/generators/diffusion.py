from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DDPMTrainingJob:
    class_id: int
    train_data_dir: Path
    output_dir: Path
    command: list[str]


def build_ddpm_command(
    train_script: Path | str,
    train_data_dir: Path | str,
    output_dir: Path | str,
    resolution: int = 32,
    train_batch_size: int = 64,
    num_epochs: int = 100,
    mixed_precision: str = "fp16",
    logger: str | None = "wandb",
) -> list[str]:
    command = [
        "accelerate",
        "launch",
        str(train_script),
        "--train_data_dir",
        str(train_data_dir),
        "--resolution",
        str(resolution),
        "--output_dir",
        str(output_dir),
        "--train_batch_size",
        str(train_batch_size),
        "--num_epochs",
        str(num_epochs),
    ]

    if mixed_precision:
        command.extend(["--mixed_precision", mixed_precision])
    if logger and logger.lower() not in {"none", "off", "disabled"}:
        command.extend(["--logger", logger])

    return command


def build_class_jobs(
    classes: tuple[int, ...] | list[int],
    data_dir: Path | str,
    output_root: Path | str,
    train_script: Path | str,
    resolution: int = 32,
    train_batch_size: int = 64,
    num_epochs: int = 100,
    mixed_precision: str = "fp16",
    logger: str | None = "wandb",
) -> list[DDPMTrainingJob]:
    jobs: list[DDPMTrainingJob] = []
    for class_id in classes:
        train_data_dir = Path(data_dir) / f"class{class_id}"
        output_dir = Path(output_root) / f"class{class_id}"
        jobs.append(
            DDPMTrainingJob(
                class_id=class_id,
                train_data_dir=train_data_dir,
                output_dir=output_dir,
                command=build_ddpm_command(
                    train_script=train_script,
                    train_data_dir=train_data_dir,
                    output_dir=output_dir,
                    resolution=resolution,
                    train_batch_size=train_batch_size,
                    num_epochs=num_epochs,
                    mixed_precision=mixed_precision,
                    logger=logger,
                ),
            )
        )
    return jobs
