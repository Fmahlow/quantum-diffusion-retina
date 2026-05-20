from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retina_synthesis.data import export_retinamnist_by_class
from retina_synthesis.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RetinaMNIST images into class folders.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/retinamnist/train"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    summary = export_retinamnist_by_class(
        output_dir=args.output_dir,
        split=args.split,
        resize=args.resize,
    )
    print(json.dumps({"output_dir": str(summary.output_dir), "split": summary.split, "counts": summary.counts}, indent=2))


if __name__ == "__main__":
    main()

