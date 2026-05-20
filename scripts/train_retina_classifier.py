from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retina_synthesis.classifier import RetinaFeatureCNN, classification_metrics
from retina_synthesis.data import count_labels, label_names, load_retinamnist, num_classes
from retina_synthesis.image_utils import collate_to_tensor01
from retina_synthesis.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a RetinaMNIST classifier for task-specific evaluation.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classifier"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weights", action="store_true")
    return parser.parse_args()


def run_epoch(
    model: RetinaFeatureCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    total_loss = 0.0
    total_items = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        logits_all.append(logits.detach().cpu())
        targets_all.append(targets.detach().cpu())

    logits_cat = torch.cat(logits_all, dim=0)
    targets_cat = torch.cat(targets_all, dim=0)
    metrics = classification_metrics(
        logits=logits_cat,
        targets=targets_cat,
        loss=total_loss / max(total_items, 1),
        num_classes=num_classes(),
    )
    return metrics.__dict__


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = load_retinamnist("train", normalize=False)
    val_dataset = load_retinamnist("val", normalize=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_to_tensor01,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_to_tensor01,
        num_workers=0,
    )

    first_images, _ = next(iter(train_loader))
    model = RetinaFeatureCNN(
        in_channels=int(first_images.shape[1]),
        num_classes=num_classes(),
        feature_dim=args.feature_dim,
    ).to(device)
    class_counts = count_labels(train_dataset)
    if args.no_class_weights:
        class_weights = None
    else:
        total = sum(class_counts.values())
        class_weights = torch.tensor(
            [total / (num_classes() * max(class_counts.get(class_id, 0), 1)) for class_id in range(num_classes())],
            dtype=torch.float32,
            device=device,
        )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    rows = []
    best_val_balanced_accuracy = -1.0
    best_path = args.output_dir / "retina_classifier.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

        if val_metrics["balanced_accuracy"] > best_val_balanced_accuracy:
            best_val_balanced_accuracy = val_metrics["balanced_accuracy"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "in_channels": model.in_channels,
                    "num_classes": model.num_classes,
                    "feature_dim": model.feature_dim,
                    "label_names": label_names(),
                    "class_counts": dict(class_counts),
                    "class_weights": None if class_weights is None else class_weights.detach().cpu().tolist(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

    pd.DataFrame(rows).to_csv(args.output_dir / "training_history.csv", index=False)
    print(f"Best classifier saved to {best_path}")


if __name__ == "__main__":
    main()
