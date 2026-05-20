from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    ordinal_mae: float


class RetinaFeatureCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 5, feature_dim: int = 128) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            self._block(in_channels, 32),
            nn.MaxPool2d(2),
            self._block(32, 64),
            nn.MaxPool2d(2),
            self._block(64, feature_dim),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return features.flatten(1)

    def forward(self, images: torch.Tensor, return_features: bool = False):
        features = self.extract_features(images)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(targets.view(-1), predictions.view(-1)):
        matrix[int(target), int(prediction)] += 1
    return matrix


def classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss: float,
    num_classes: int,
) -> ClassificationMetrics:
    predictions = logits.argmax(dim=1)
    matrix = confusion_matrix(predictions.cpu(), targets.cpu(), num_classes=num_classes).float()

    accuracy = float((predictions == targets).float().mean().item())
    recalls = matrix.diag() / matrix.sum(dim=1).clamp_min(1.0)
    precision = matrix.diag() / matrix.sum(dim=0).clamp_min(1.0)
    f1 = 2.0 * precision * recalls / (precision + recalls).clamp_min(1e-12)
    ordinal_mae = (predictions.float() - targets.float()).abs().mean().item()

    return ClassificationMetrics(
        loss=float(loss),
        accuracy=accuracy,
        balanced_accuracy=float(recalls.mean().item()),
        macro_f1=float(f1.mean().item()),
        ordinal_mae=float(ordinal_mae),
    )

