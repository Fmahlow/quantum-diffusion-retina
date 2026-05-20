from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from diffusers import DDPMPipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retina_synthesis.data import label_names, load_retinamnist
from retina_synthesis.image_utils import any_to_tensor01
from retina_synthesis.reproducibility import set_seed
from retina_synthesis.task_metrics import (
    classifier_outputs,
    collect_class_images,
    feature_fid,
    load_feature_classifier,
    mean_nearest_neighbor_distance,
    mmd_rbf,
    resize_like,
    within_set_nn_distance,
)


def parse_classes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DDPM samples with a RetinaMNIST-trained classifier.")
    parser.add_argument("--pipeline-root", type=Path, default=Path("outputs/ddpm"))
    parser.add_argument("--classifier-checkpoint", type=Path, default=Path("outputs/classifier/retina_classifier.pt"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--classes", type=parse_classes, default=parse_classes("0,1,2,3,4"))
    parser.add_argument("--num-generated", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/ddpm_task_metrics.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def generate_images(
    pipeline_dir: Path,
    num_images: int,
    device: str,
    dtype: torch.dtype,
    num_inference_steps: int,
    seed: int,
) -> torch.Tensor:
    pipe = DDPMPipeline.from_pretrained(str(pipeline_dir), torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(seed)
    images = []

    remaining = num_images
    while remaining > 0:
        batch_size = min(remaining, 64)
        batch = pipe(
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images
        images.extend(any_to_tensor01(image) for image in batch)
        remaining -= batch_size

    return torch.stack(images, dim=0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.no_fp16 or device == "cpu" else torch.float16

    dataset = load_retinamnist(args.split, normalize=False)
    classifier = load_feature_classifier(args.classifier_checkpoint, device=device)
    labels = label_names()

    rows = []
    for class_id in args.classes:
        real_images = collect_class_images(dataset, class_id=class_id, limit=args.num_generated)
        fake_images = generate_images(
            pipeline_dir=args.pipeline_root / f"class{class_id}",
            num_images=args.num_generated,
            device=device,
            dtype=dtype,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed + class_id,
        )
        fake_images = resize_like(fake_images, real_images)

        real_logits, real_features = classifier_outputs(classifier, real_images, device=device, batch_size=args.batch_size)
        fake_logits, fake_features = classifier_outputs(classifier, fake_images, device=device, batch_size=args.batch_size)

        fake_probs = fake_logits.softmax(dim=1)
        fake_predictions = fake_probs.argmax(dim=1)
        real_predictions = real_logits.argmax(dim=1)
        target = torch.full_like(fake_predictions, fill_value=class_id)
        real_target = torch.full_like(real_predictions, fill_value=class_id)

        real_nn = within_set_nn_distance(real_features)
        fake_nn = within_set_nn_distance(fake_features)

        rows.append(
            {
                "Model": "DDPM",
                "Generator_Label": labels.get(class_id, f"class{class_id}"),
                "Real_Label": labels.get(class_id, f"class{class_id}"),
                "Class_ID": class_id,
                "N_Real": int(real_images.size(0)),
                "N_Fake": int(fake_images.size(0)),
                "Feature_FID": feature_fid(real_features, fake_features),
                "Feature_MMD_RBF": mmd_rbf(real_features, fake_features),
                "Fake_Target_Accuracy": float((fake_predictions == target).float().mean().item()),
                "Fake_Target_Confidence": float(fake_probs[:, class_id].mean().item()),
                "Fake_Prediction_MAE": float((fake_predictions.float() - target.float()).abs().mean().item()),
                "Real_Target_Accuracy": float((real_predictions == real_target).float().mean().item()),
                "Fake_To_Real_NN_Distance": mean_nearest_neighbor_distance(fake_features, real_features),
                "Real_Within_NN_Distance": real_nn,
                "Fake_Within_NN_Distance": fake_nn,
                "Fake_Diversity_Ratio": float(fake_nn / real_nn) if real_nn == real_nn and real_nn > 0 else float("nan"),
            }
        )

    results = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    print(results)


if __name__ == "__main__":
    main()

