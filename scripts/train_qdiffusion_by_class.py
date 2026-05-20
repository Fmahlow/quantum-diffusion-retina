"""Train one hybrid quantum DDPM per RetinaMNIST class.

Usage:
    python scripts/train_qdiffusion_by_class.py
    python scripts/train_qdiffusion_by_class.py --config configs/qdiffusion_retinamnist.json
    python scripts/train_qdiffusion_by_class.py --classes 0 1 --n-epochs 100 --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retina_synthesis.data import label_names, load_retinamnist
from retina_synthesis.image_utils import any_to_tensor01
from retina_synthesis.reproducibility import set_seed
from retina_synthesis.quantum.qdiffusion import QDiffusionConfig, train_qdiffusion_for_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one hybrid quantum DDPM per RetinaMNIST class.")
    parser.add_argument("--config", type=Path, default=Path("configs/qdiffusion_retinamnist.json"))
    parser.add_argument("--classes", type=int, nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--n-quantum-layers", type=int, default=None)
    parser.add_argument("--base-ch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu/cuda). Defaults to cuda if available.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_class_images(dataset, class_id: int, img_size: int) -> torch.Tensor:
    """Collect all images for a class as float32 [0,1] tensors."""
    images: list[torch.Tensor] = []
    for image, label in dataset:
        label_id = int(label.long().view(-1)[0]) if torch.is_tensor(label) else int(label)
        if label_id != class_id:
            continue
        t = any_to_tensor01(image)
        if t.shape[-2] != img_size or t.shape[-1] != img_size:
            t = F.interpolate(
                t.unsqueeze(0), size=(img_size, img_size), mode="bilinear", align_corners=False
            ).squeeze(0)
        images.append(t)
    if not images:
        raise ValueError(f"No images found for class {class_id}.")
    return torch.stack(images, dim=0)


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg_dict = json.load(f)

    classes = args.classes if args.classes is not None else cfg_dict.get("classes", [0, 1, 2, 3, 4])
    output_root = Path(args.output_root or cfg_dict.get("output_root", "outputs/qdiffusion"))

    config = QDiffusionConfig(
        n_qubits=args.n_qubits or cfg_dict.get("n_qubits", 8),
        n_quantum_layers=args.n_quantum_layers or cfg_dict.get("n_quantum_layers", 3),
        img_channels=cfg_dict.get("img_channels", 3),
        img_size=cfg_dict.get("img_size", 32),
        base_ch=args.base_ch or cfg_dict.get("base_ch", 64),
        t_dim=cfg_dict.get("t_dim", 256),
        emb_dim=cfg_dict.get("emb_dim", 1024),
        dropout=cfg_dict.get("dropout", 0.1),
        T=cfg_dict.get("T", 1000),
        n_epochs=args.n_epochs or cfg_dict.get("n_epochs", 300),
        batch_size=args.batch_size or cfg_dict.get("batch_size", 32),
        lr=cfg_dict.get("lr", 2e-4),
        grad_clip=cfg_dict.get("grad_clip", 1.0),
        seed=args.seed or cfg_dict.get("seed", 42),
        save_every=cfg_dict.get("save_every", 100),
        inference_stride=cfg_dict.get("inference_stride", 10),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "qdiffusion_config_frozen.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    set_seed(config.seed)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    split = cfg_dict.get("split_for_training", "train")
    dataset = load_retinamnist(split, normalize=False)
    names = label_names()

    print(f"Training Hybrid Quantum DDPM for classes {classes} on split='{split}'")
    print(f"Config: n_qubits={config.n_qubits}, n_layers={config.n_quantum_layers}, "
          f"base_ch={config.base_ch}, T={config.T}, "
          f"n_epochs={config.n_epochs}, batch_size={config.batch_size}")
    print(f"Device: {device}\n")

    for class_id in classes:
        print(f"=== Class {class_id} ({names.get(class_id, '?')}) ===")
        real_images = load_class_images(dataset, class_id=class_id, img_size=config.img_size)
        print(f"  {real_images.size(0)} training images, shape {tuple(real_images.shape[1:])}")

        class_output_dir = output_root / f"class{class_id}"
        train_qdiffusion_for_class(
            real_images_01=real_images,
            class_id=class_id,
            output_dir=class_output_dir,
            config=config,
            device=device,
            verbose=not args.quiet,
        )
        print(f"  Saved to {class_output_dir}\n")

    print("All classes done.")


if __name__ == "__main__":
    main()
