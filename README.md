# Quantum Diffusion Retina

Research code for synthetic RetinaMNIST experiments with classical diffusion baselines and planned quantum / hybrid quantum-classical generative models.

The original exploratory notebook is preserved at:

```text
diffusion_classical_retinamnist (1).ipynb
```

Reusable experiment code now lives under `src/retina_synthesis`, with command-line entry points in `scripts`.

## What is included

- RetinaMNIST export and preprocessing utilities.
- Class-specific DDPM training command generation.
- DDPM evaluation with FID and Inception Score.
- Task-specific evaluation using a RetinaMNIST classifier trained only on real data.
- Feature-space metrics, target-class fidelity, nearest-neighbor diversity checks, and report-ready plots.
- Scientific planning notes for quantum synthetic RetinaMNIST experiments.

## Project layout

```text
configs/      Frozen experiment parameters
docs/         Scientific plan and execution guides
notebooks/    Notebook workflow notes
scripts/      Reproducible command-line entry points
src/          Reusable Python package code
reports/      Generated report artifacts, ignored except .gitkeep
```

## Setup

Use one Python environment consistently. The notebook metadata was created with Python 3.11, while the local shell may point to another environment.

```bash
python -c "import torch, torchvision, diffusers, accelerate, medmnist, pandas, matplotlib; print(torch.__version__)"
```

Install core dependencies:

```bash
pip install -r requirements.txt
```

For DDPM training, clone the Hugging Face Diffusers repository and install its training extras:

```bash
git clone https://github.com/huggingface/diffusers.git
pip install -e "diffusers[training]"
pip install -r diffusers/examples/unconditional_image_generation/requirements.txt
```

## Running experiments

See [docs/run_experiments.md](docs/run_experiments.md) for the full workflow.

High-level sequence:

```bash
python scripts/export_retinamnist.py --split train --output-dir data/retinamnist/train --resize 32
python scripts/train_ddpm_by_class.py --data-dir data/retinamnist/train --output-root outputs/ddpm
python scripts/evaluate_ddpm.py --pipeline-root outputs/ddpm --split test --output-csv outputs/ddpm_metrics.csv
python scripts/train_retina_classifier.py --output-dir outputs/classifier
python scripts/evaluate_ddpm_task_metrics.py --pipeline-root outputs/ddpm --output-csv outputs/ddpm_task_metrics.csv
python scripts/combine_metric_csvs.py --inputs outputs/ddpm_metrics.csv outputs/ddpm_task_metrics.csv --output-csv outputs/ddpm_all_metrics.csv
python scripts/plot_metrics.py --metrics-csv outputs/ddpm_all_metrics.csv --output-dir reports/ddpm_all
```

## Evaluation protocol

FID and Inception Score are included for comparability, but the main evaluation should use target-domain evidence:

- RetinaMNIST classifier feature FID.
- Feature-space MMD.
- Target-class accuracy and confidence of generated images.
- Ordinal prediction error.
- Nearest-neighbor diversity and memorization checks.
- Downstream augmentation experiments across multiple seeds.

See [docs/evaluation_protocol.md](docs/evaluation_protocol.md).

## Quantum model plan

The planned quantum baseline is a hybrid quantum-classical GAN or QCBM-latent model. See [docs/quantum_synthetic_retinamnist_plan.md](docs/quantum_synthetic_retinamnist_plan.md).

