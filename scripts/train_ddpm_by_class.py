from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retina_synthesis.generators.diffusion import build_class_jobs
from retina_synthesis.reproducibility import set_seed


def parse_classes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one unconditional DDPM per RetinaMNIST class.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/retinamnist/train"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ddpm"))
    parser.add_argument("--train-script", type=Path, default=Path("diffusers/examples/unconditional_image_generation/train_unconditional.py"))
    parser.add_argument("--classes", type=parse_classes, default=parse_classes("0,1,2,3,4"))
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mixed-precision", default="fp16")
    parser.add_argument("--logger", default="wandb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    jobs = build_class_jobs(
        classes=args.classes,
        data_dir=args.data_dir,
        output_root=args.output_root,
        train_script=args.train_script,
        resolution=args.resolution,
        train_batch_size=args.batch_size,
        num_epochs=args.epochs,
        mixed_precision=args.mixed_precision,
        logger=args.logger,
    )

    for job in jobs:
        print(shlex.join(job.command), flush=True)
        if not args.dry_run:
            subprocess.run(job.command, check=True)


if __name__ == "__main__":
    main()

