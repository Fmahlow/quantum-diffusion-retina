from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from diffusers import DDPMPipeline
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retina_synthesis.reproducibility import set_seed


def parse_classes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sample grids from class-specific DDPM pipelines.")
    parser.add_argument("--pipeline-root", type=Path, default=Path("outputs/ddpm"))
    parser.add_argument("--classes", type=parse_classes, default=parse_classes("0,1,2,3,4"))
    parser.add_argument("--num-samples", type=int, default=25)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/ddpm/samples"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def make_grid(images: list[Image.Image], title: str, cols: int | None = None, padding: int = 4) -> Image.Image:
    if not images:
        raise ValueError("Cannot create a grid with no images.")

    cols = cols or int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))
    width, height = images[0].size
    title_height = 22

    canvas = Image.new(
        "RGB",
        (
            cols * width + (cols + 1) * padding,
            rows * height + (rows + 1) * padding + title_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, padding), title, fill="black")

    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        x = padding + col * (width + padding)
        y = title_height + padding + row * (height + padding)
        canvas.paste(image.convert("RGB"), (x, y))

    return canvas


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.no_fp16 or device == "cpu" else torch.float16

    for class_id in args.classes:
        pipeline_dir = args.pipeline_root / f"class{class_id}"
        pipe = DDPMPipeline.from_pretrained(str(pipeline_dir), torch_dtype=dtype).to(device)
        pipe.set_progress_bar_config(disable=True)
        generator = torch.Generator(device=device).manual_seed(args.seed + class_id)
        images = pipe(
            batch_size=args.num_samples,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images
        grid = make_grid(images, title=f"DDPM class {class_id}")
        grid.save(args.output_dir / f"class{class_id}_samples.png")
        print(f"Wrote {args.output_dir / f'class{class_id}_samples.png'}")


if __name__ == "__main__":
    main()

