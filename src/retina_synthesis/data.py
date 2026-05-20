from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchvision.transforms as transforms
from medmnist import INFO, RetinaMNIST
from PIL import Image


DATA_FLAG = "retinamnist"


@dataclass(frozen=True)
class ExportSummary:
    output_dir: Path
    split: str
    counts: dict[int, int]


def label_names() -> dict[int, str]:
    labels = INFO[DATA_FLAG]["label"]
    return {int(key): value for key, value in labels.items()}


def num_classes() -> int:
    return len(label_names())


def image_transform(normalize: bool = False) -> transforms.Compose:
    steps: list[object] = [transforms.ToTensor()]
    if normalize:
        steps.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    return transforms.Compose(steps)


def load_retinamnist(split: str, normalize: bool = False, download: bool = True) -> RetinaMNIST:
    return RetinaMNIST(split=split, transform=image_transform(normalize), download=download)


def count_labels(dataset: Iterable[tuple[object, object]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for _, label in dataset:
        if torch.is_tensor(label):
            label_id = int(label.long().view(-1)[0])
        else:
            label_id = int(np.asarray(label).reshape(-1)[0])
        counts[label_id] += 1
    return counts


def array_to_image(image: np.ndarray) -> Image.Image:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return Image.fromarray(arr)


def export_retinamnist_by_class(
    output_dir: Path | str = Path("data/retinamnist/train"),
    split: str = "train",
    resize: int | None = None,
    download: bool = True,
) -> ExportSummary:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = RetinaMNIST(split=split, download=download)
    labels = np.asarray(dataset.labels).reshape(-1)
    counts: Counter[int] = Counter()

    for class_id in range(num_classes()):
        (output_path / f"class{class_id}").mkdir(parents=True, exist_ok=True)

    for index, (image_array, label) in enumerate(zip(dataset.imgs, labels)):
        label_id = int(label)
        image = array_to_image(image_array)
        if resize is not None:
            image = image.resize((resize, resize), Image.BILINEAR)
        image.save(output_path / f"class{label_id}" / f"{index:05d}.png")
        counts[label_id] += 1

    return ExportSummary(output_dir=output_path, split=split, counts=dict(sorted(counts.items())))

