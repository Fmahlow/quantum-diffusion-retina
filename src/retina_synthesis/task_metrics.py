from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from retina_synthesis.classifier import RetinaFeatureCNN
from retina_synthesis.image_utils import any_to_tensor01


def load_feature_classifier(checkpoint_path: str | Path, device: str) -> RetinaFeatureCNN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RetinaFeatureCNN(
        in_channels=int(checkpoint["in_channels"]),
        num_classes=int(checkpoint["num_classes"]),
        feature_dim=int(checkpoint.get("feature_dim", 128)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def collect_class_images(dataset, class_id: int, limit: int | None = None) -> torch.Tensor:
    images: list[torch.Tensor] = []
    for image, label in dataset:
        label_id = int(label.long().view(-1)[0]) if torch.is_tensor(label) else int(label)
        if label_id != class_id:
            continue
        images.append(any_to_tensor01(image))
        if limit is not None and len(images) >= limit:
            break
    if not images:
        raise ValueError(f"No images found for class {class_id}.")
    return torch.stack(images, dim=0)


def resize_like(images: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if images.shape[-2:] == reference.shape[-2:]:
        return images
    return F.interpolate(images, size=reference.shape[-2:], mode="bilinear", align_corners=False)


@torch.no_grad()
def classifier_outputs(
    model: RetinaFeatureCNN,
    images: torch.Tensor,
    device: str,
    batch_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_list: list[torch.Tensor] = []
    features_list: list[torch.Tensor] = []
    for start in range(0, images.size(0), batch_size):
        batch = images[start : start + batch_size].to(device)
        logits, features = model(batch, return_features=True)
        logits_list.append(logits.cpu())
        features_list.append(features.cpu())
    return torch.cat(logits_list, dim=0), torch.cat(features_list, dim=0)


def covariance(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    denom = max(features.size(0) - 1, 1)
    return centered.T @ centered / denom


def matrix_sqrt_psd(matrix: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh((matrix + matrix.T) * 0.5)
    values = values.clamp_min(0.0).sqrt()
    return (vectors * values.unsqueeze(0)) @ vectors.T


def feature_fid(real_features: torch.Tensor, fake_features: torch.Tensor, eps: float = 1e-6) -> float:
    real_features = real_features.double()
    fake_features = fake_features.double()

    mu_real = real_features.mean(dim=0)
    mu_fake = fake_features.mean(dim=0)
    sigma_real = covariance(real_features)
    sigma_fake = covariance(fake_features)

    eye = torch.eye(sigma_real.size(0), dtype=torch.double)
    sigma_real = sigma_real + eps * eye
    sigma_fake = sigma_fake + eps * eye

    sqrt_real = matrix_sqrt_psd(sigma_real)
    cov_mean = matrix_sqrt_psd(sqrt_real @ sigma_fake @ sqrt_real)
    score = (mu_real - mu_fake).pow(2).sum() + torch.trace(sigma_real + sigma_fake - 2.0 * cov_mean)
    return float(score.clamp_min(0.0).item())


def mmd_rbf(real_features: torch.Tensor, fake_features: torch.Tensor, max_samples: int = 512) -> float:
    real = real_features.float()
    fake = fake_features.float()
    if real.size(0) > max_samples:
        real = real[:max_samples]
    if fake.size(0) > max_samples:
        fake = fake[:max_samples]

    if real.size(0) < 2 or fake.size(0) < 2:
        return float("nan")

    combined = torch.cat([real, fake], dim=0)
    distances = torch.cdist(combined, combined).pow(2)
    positive = distances[distances > 0]
    bandwidth = positive.median().clamp_min(1e-12)

    k_xx = torch.exp(-torch.cdist(real, real).pow(2) / (2.0 * bandwidth))
    k_yy = torch.exp(-torch.cdist(fake, fake).pow(2) / (2.0 * bandwidth))
    k_xy = torch.exp(-torch.cdist(real, fake).pow(2) / (2.0 * bandwidth))

    n = real.size(0)
    m = fake.size(0)
    xx = (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1))
    yy = (k_yy.sum() - k_yy.diag().sum()) / (m * (m - 1))
    xy = k_xy.mean()
    return float((xx + yy - 2.0 * xy).item())


def mean_nearest_neighbor_distance(source_features: torch.Tensor, reference_features: torch.Tensor) -> float:
    distances = torch.cdist(source_features.float(), reference_features.float())
    return float(distances.min(dim=1).values.mean().item())


def within_set_nn_distance(features: torch.Tensor) -> float:
    if features.size(0) < 2:
        return float("nan")
    distances = torch.cdist(features.float(), features.float())
    distances.fill_diagonal_(float("inf"))
    return float(distances.min(dim=1).values.mean().item())

