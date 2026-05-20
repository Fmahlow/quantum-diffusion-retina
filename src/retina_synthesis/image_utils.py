from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def pil_to_tensor01(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    tensor = torch.from_numpy(arr).float() / 255.0
    return tensor.permute(2, 0, 1)


def any_to_tensor01(value: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(value, Image.Image):
        tensor = pil_to_tensor01(value)
    elif isinstance(value, np.ndarray):
        arr = value
        if arr.ndim == 2:
            arr = arr[:, :, None]
        tensor = torch.from_numpy(arr).float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        else:
            tensor = tensor.unsqueeze(0)
    elif torch.is_tensor(value):
        tensor = value.detach().float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.min() < 0.0:
            tensor = (tensor + 1.0) * 0.5
        elif tensor.max() > 1.0:
            tensor = tensor / 255.0
    else:
        raise TypeError(f"Unsupported image type: {type(value)!r}")

    return tensor.clamp(0.0, 1.0)


def to_uint8_3ch_299(tensor01: torch.Tensor) -> torch.Tensor:
    if tensor01.shape[0] == 1:
        tensor01 = tensor01.repeat(3, 1, 1)
    elif tensor01.shape[0] > 3:
        tensor01 = tensor01[:3]

    batch = tensor01.unsqueeze(0)
    batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
    return (batch.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def collate_to_tensor01(batch: list[tuple[object, object]]) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = zip(*batch)
    image_tensor = torch.stack([any_to_tensor01(image) for image in images], dim=0)

    label_values: list[torch.Tensor] = []
    for label in labels:
        if torch.is_tensor(label):
            label_values.append(label.long().view(-1)[0])
        else:
            label_values.append(torch.tensor(int(np.asarray(label).reshape(-1)[0]), dtype=torch.long))

    return image_tensor, torch.stack(label_values, dim=0)

