"""Evaluate per-class hybrid quantum DDPM generators with RetinaMNIST task metrics.

Produces the same metric columns as evaluate_ddpm_task_metrics.py so both
CSVs can be stacked directly for classical vs quantum comparison.

Usage:
    python scripts/evaluate_qdiffusion_task_metrics.py
    python scripts/evaluate_qdiffusion_task_metrics.py \\
        --qdiffusion-root outputs/qdiffusion \\
        --classifier-checkpoint outputs/classifier/retina_classifier.pt \\
        --output-csv outputs/qdiffusion_task_metrics.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retina_synthesis.data import label_names, load_retinamnist
from retina_synthesis.reproducibility import set_seed
from retina_synthesis.task_metrics import (
    classifier_outputs,
    collect_class_images,
    feature_fid,
    load_feature_classifier,
    mean_nearest_neighbor_distance,
    mmd_rbf,
    within_set_nn_distance,
)
from retina_synthesis.quantum.qdiffusion import load_model_from_checkpoint, sample_qdiffusion


def parse_classes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate QDiffusion samples with a RetinaMNIST-trained classifier."
    )
    parser.add_argument("--qdiffusion-root", type=Path, default=Path("outputs/qdiffusion"))
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=Path("outputs/classifier/retina_classifier.pt"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--classes", type=parse_classes, default=parse_classes("0,1,2,3,4"))
    parser.add_argument("--num-generated", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/qdiffusion_task_metrics.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_retinamnist(args.split, normalize=False)
    classifier = load_feature_classifier(args.classifier_checkpoint, device=device)
    names = label_names()

    rows = []
    for class_id in args.classes:
        checkpoint_path = args.qdiffusion_root / f"class{class_id}" / "final.pt"
        if not checkpoint_path.exists():
            print(f"[WARNING] No checkpoint at {checkpoint_path}, skipping class {class_id}.")
            continue

        print(f"Evaluating class {class_id} ({names.get(class_id, '?')})...")

        model, config = load_model_from_checkpoint(checkpoint_path, device=device)

        real_images = collect_class_images(dataset, class_id=class_id, limit=args.num_generated)
        fake_images = sample_qdiffusion(
            model=model,
            config=config,
            n_samples=args.num_generated,
            device=device,
            seed=args.seed + class_id,
        )

        if fake_images.shape[-2:] != real_images.shape[-2:]:
            fake_images = F.interpolate(
                fake_images,
                size=real_images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        real_logits, real_features = classifier_outputs(
            classifier, real_images, device=device, batch_size=args.batch_size
        )
        fake_logits, fake_features = classifier_outputs(
            classifier, fake_images, device=device, batch_size=args.batch_size
        )

        fake_probs = fake_logits.softmax(dim=1)
        fake_predictions = fake_probs.argmax(dim=1)
        real_predictions = real_logits.argmax(dim=1)
        target = torch.full_like(fake_predictions, fill_value=class_id)
        real_target = torch.full_like(real_predictions, fill_value=class_id)

        real_nn = within_set_nn_distance(real_features)
        fake_nn = within_set_nn_distance(fake_features)

        rows.append(
            {
                "Model": "QDiffusion",
                "Generator_Label": names.get(class_id, f"class{class_id}"),
                "Real_Label": names.get(class_id, f"class{class_id}"),
                "Class_ID": class_id,
                "N_Real": int(real_images.size(0)),
                "N_Fake": int(fake_images.size(0)),
                "Feature_FID": feature_fid(real_features, fake_features),
                "Feature_MMD_RBF": mmd_rbf(real_features, fake_features),
                "Fake_Target_Accuracy": float((fake_predictions == target).float().mean().item()),
                "Fake_Target_Confidence": float(fake_probs[:, class_id].mean().item()),
                "Fake_Prediction_MAE": float(
                    (fake_predictions.float() - target.float()).abs().mean().item()
                ),
                "Real_Target_Accuracy": float((real_predictions == real_target).float().mean().item()),
                "Fake_To_Real_NN_Distance": mean_nearest_neighbor_distance(fake_features, real_features),
                "Real_Within_NN_Distance": real_nn,
                "Fake_Within_NN_Distance": fake_nn,
                "Fake_Diversity_Ratio": (
                    float(fake_nn / real_nn) if real_nn == real_nn and real_nn > 0 else float("nan")
                ),
            }
        )

    if not rows:
        print("No classes evaluated. Check that checkpoints exist in --qdiffusion-root.")
        return

    results = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
