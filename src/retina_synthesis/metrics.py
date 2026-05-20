from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from diffusers import DDPMPipeline
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from retina_synthesis.image_utils import any_to_tensor01, collate_to_tensor01, to_uint8_3ch_299


@torch.no_grad()
def evaluate_diffusion_pipeline_for_class(
    pipeline_dir: str | Path,
    device: str,
    dataset,
    label_target: int,
    generator_label_name: str,
    real_label_name: str,
    batch_size: int = 64,
    max_batches: int = 10,
    num_inference_steps: int = 50,
    use_fp16: bool = True,
) -> dict[str, float | str]:
    fid = FrechetInceptionDistance(feature=64).to(device)
    inception_score = InceptionScore().to(device)

    pipe = DDPMPipeline.from_pretrained(
        str(pipeline_dir),
        torch_dtype=(torch.float16 if use_fp16 else torch.float32),
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_to_tensor01,
        num_workers=0,
    )

    batch_count = 0
    for real, labels in loader:
        mask = labels == label_target
        if mask.sum().item() == 0:
            continue

        real_selected = real[mask]
        generated = pipe(
            batch_size=int(real_selected.size(0)),
            num_inference_steps=num_inference_steps,
        ).images

        real_uint8 = torch.cat([to_uint8_3ch_299(item) for item in real_selected], dim=0).to(device)
        fake_uint8 = torch.cat([to_uint8_3ch_299(any_to_tensor01(item)) for item in generated], dim=0).to(device)

        fid.update(real_uint8, real=True)
        fid.update(fake_uint8, real=False)
        inception_score.update(fake_uint8)

        batch_count += 1
        if batch_count >= max_batches:
            break

    is_mean, is_std = inception_score.compute()
    return {
        "Model": "DDPM",
        "Generator_Label": generator_label_name,
        "Real_Label": real_label_name,
        "FID": float(fid.compute().item()),
        "IS_Mean": float(is_mean.item() if torch.is_tensor(is_mean) else is_mean),
        "IS_Std": float(is_std.item() if torch.is_tensor(is_std) else is_std),
    }


def evaluate_diffusion_pipelines(
    pipeline_dirs: dict[int, str | Path],
    device: str,
    dataset,
    label_names: dict[int, str] | None = None,
    batch_size: int = 64,
    max_batches: int = 10,
    num_inference_steps: int = 50,
    use_fp16: bool = True,
) -> pd.DataFrame:
    rows = []
    for class_id, pipeline_dir in sorted(pipeline_dirs.items()):
        label_name = label_names.get(class_id, f"class{class_id}") if label_names else f"class{class_id}"
        rows.append(
            evaluate_diffusion_pipeline_for_class(
                pipeline_dir=pipeline_dir,
                device=device,
                dataset=dataset,
                label_target=class_id,
                generator_label_name=label_name,
                real_label_name=label_name,
                batch_size=batch_size,
                max_batches=max_batches,
                num_inference_steps=num_inference_steps,
                use_fp16=use_fp16,
            )
        )
    return pd.DataFrame(rows)

