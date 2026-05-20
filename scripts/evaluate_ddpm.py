from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retina_synthesis.data import label_names, load_retinamnist
from retina_synthesis.metrics import evaluate_diffusion_pipelines
from retina_synthesis.reproducibility import set_seed


def parse_classes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate class-specific DDPM pipelines on RetinaMNIST.")
    parser.add_argument("--pipeline-root", type=Path, default=Path("outputs/ddpm"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--classes", type=parse_classes, default=parse_classes("0,1,2,3,4"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/ddpm_metrics.csv"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_retinamnist(split=args.split, normalize=False)
    pipeline_dirs = {class_id: args.pipeline_root / f"class{class_id}" for class_id in args.classes}
    results = evaluate_diffusion_pipelines(
        pipeline_dirs=pipeline_dirs,
        device=device,
        dataset=dataset,
        label_names=label_names(),
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        num_inference_steps=args.num_inference_steps,
        use_fp16=not args.no_fp16,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    print(results)


if __name__ == "__main__":
    main()

