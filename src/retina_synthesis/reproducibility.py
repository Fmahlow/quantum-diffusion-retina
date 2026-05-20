from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

try:
    import torch
except Exception:
    torch = None


@dataclass(frozen=True)
class SeedState:
    seed: int
    deterministic: bool


def set_seed(seed: int = 42, deterministic: bool = True) -> SeedState:
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return SeedState(seed=seed, deterministic=False)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    return SeedState(seed=seed, deterministic=deterministic)
