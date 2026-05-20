# Running experiments — classical DDPM vs hybrid quantum DDPM

This document is the single authoritative guide for reproducing the full
classical vs quantum comparison on RetinaMNIST.  Steps 1–5 cover the
classical DDPM baseline.  Steps 6–7 cover the hybrid quantum DDPM
(QDiffusion).  Steps 8–9 cover evaluation and comparison.

---

## 1. Environment

Use the same Python environment for all steps (Python 3.10+ recommended).
Verify that the core imports work before proceeding:

```bash
python -c "import torch, torchvision, diffusers, accelerate, medmnist, pennylane; \
           print('torch', torch.__version__, '| pennylane', pennylane.__version__)"
```

Install all dependencies from the project root:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
# Core
pip install torch torchvision diffusers accelerate medmnist datasets
pip install pandas matplotlib "torchmetrics[image]" torch-fidelity

# Quantum simulation
pip install pennylane pennylane-lightning
```

For the classical DDPM training script, the Hugging Face diffusers training
examples are also required:

```bash
git clone https://github.com/huggingface/diffusers.git
pip install -e "diffusers[training]"
pip install -r diffusers/examples/unconditional_image_generation/requirements.txt
```

If using Weights & Biases:

```bash
wandb login
```

Pass `--logger tensorboard` or omit the flag entirely to skip W&B.

---

## 2. Export RetinaMNIST by class

This only needs to run once.  It downloads the dataset and writes one
folder per class that the classical DDPM training script expects.

```bash
python scripts/export_retinamnist.py \
  --split train \
  --output-dir data/retinamnist/train \
  --resize 32
```

Expected output:

```
data/retinamnist/train/
  class0/   (~216 images, DR grade 0)
  class1/   (~216 images, DR grade 1)
  class2/   (~216 images, DR grade 2)
  class3/   (~216 images, DR grade 3)
  class4/   (~216 images, DR grade 4)
```

---

## 3. Train the classifier evaluator

Train a `RetinaFeatureCNN` on the **real** training split only.  This model
is not a generator; it is a fixed evaluation instrument used in all
task-specific metrics.  It must be trained before any generative model
evaluation.

```bash
python scripts/train_retina_classifier.py \
  --output-dir outputs/classifier \
  --epochs 30 \
  --batch-size 64 \
  --seed 42
```

Class-imbalance weights are enabled by default.  The best checkpoint is
saved to `outputs/classifier/retina_classifier.pt`.

---

## 4. Train the classical DDPM baseline

### Dry run (inspect commands only)

```bash
python scripts/train_ddpm_by_class.py --dry-run
```

### Full training

```bash
python scripts/train_ddpm_by_class.py \
  --data-dir data/retinamnist/train \
  --output-root outputs/ddpm \
  --classes 0,1,2,3,4 \
  --resolution 32 \
  --batch-size 64 \
  --epochs 100 \
  --mixed-precision fp16 \
  --logger wandb
```

### Quick smoke test (single class, 1 epoch)

```bash
python scripts/train_ddpm_by_class.py \
  --classes 0 \
  --epochs 1 \
  --logger tensorboard
```

Checkpoints are saved to `outputs/ddpm/class{0..4}/`.

---

## 5. Generate DDPM sample grids

```bash
python scripts/sample_ddpm_grid.py \
  --pipeline-root outputs/ddpm \
  --classes 0,1,2,3,4 \
  --num-samples 25 \
  --num-inference-steps 50 \
  --output-dir reports/ddpm/samples
```

Add `--no-fp16` for CPU-only environments.  Outputs one PNG grid per class
in `reports/ddpm/samples/`.

---

## 6. Train the hybrid quantum DDPM (QDiffusion)

The QDiffusion model is a standard DDPM U-Net for 32×32 images where the
bottleneck (4×4 spatial) replaces classical channel attention with a
parameterized quantum circuit: 8-qubit `StronglyEntanglingLayers` that acts
as a squeeze-and-excite gate.  One model is trained per class.

All training runs on the device returned by `torch.cuda.is_available()`;
the quantum layer uses PennyLane's `default.qubit` with a PyTorch backend
and follows the tensor device automatically.

### Full training (all classes, default config)

```bash
python scripts/train_qdiffusion_by_class.py \
  --config configs/qdiffusion_retinamnist.json
```

### Custom run (override specific parameters)

```bash
python scripts/train_qdiffusion_by_class.py \
  --config configs/qdiffusion_retinamnist.json \
  --classes 0 1 2 \
  --n-epochs 200 \
  --batch-size 32 \
  --seed 7
```

### Force CPU (e.g. no GPU available)

```bash
python scripts/train_qdiffusion_by_class.py \
  --config configs/qdiffusion_retinamnist.json \
  --device cpu
```

### Quick smoke test (single class, 5 epochs)

```bash
python scripts/train_qdiffusion_by_class.py \
  --classes 0 \
  --n-epochs 5 \
  --quiet
```

Checkpoints are saved to `outputs/qdiffusion/class{0..4}/`:

```
outputs/qdiffusion/
  qdiffusion_config_frozen.json   ← frozen hyperparameters for this run
  class0/
    checkpoint_epoch0100.pt
    checkpoint_epoch0200.pt
    checkpoint_epoch0300.pt
    final.pt                      ← used by evaluation scripts
  class1/
    ...
```

Key hyperparameters (see `configs/qdiffusion_retinamnist.json`):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_qubits` | 8 | Qubits in the bottleneck circuit |
| `n_quantum_layers` | 3 | StronglyEntanglingLayers depth |
| `base_ch` | 64 | U-Net base channel width |
| `T` | 1000 | Diffusion timesteps |
| `n_epochs` | 300 | Training epochs per class |
| `batch_size` | 32 | |
| `inference_stride` | 10 | Steps skipped during sampling (100 effective steps) |

---

## 7. Evaluate generative models

### 7a. Evaluate classical DDPM — FID / Inception Score

```bash
python scripts/evaluate_ddpm.py \
  --pipeline-root outputs/ddpm \
  --split test \
  --classes 0,1,2,3,4 \
  --batch-size 64 \
  --max-batches 10 \
  --num-inference-steps 50 \
  --output-csv outputs/ddpm_metrics.csv
```

Add `--no-fp16` on CPU.

### 7b. Evaluate classical DDPM — task-specific metrics

```bash
python scripts/evaluate_ddpm_task_metrics.py \
  --pipeline-root outputs/ddpm \
  --classifier-checkpoint outputs/classifier/retina_classifier.pt \
  --split test \
  --classes 0,1,2,3,4 \
  --num-generated 200 \
  --output-csv outputs/ddpm_task_metrics.csv
```

### 7c. Evaluate QDiffusion — task-specific metrics

```bash
python scripts/evaluate_qdiffusion_task_metrics.py \
  --qdiffusion-root outputs/qdiffusion \
  --classifier-checkpoint outputs/classifier/retina_classifier.pt \
  --split test \
  --classes 0,1,2,3,4 \
  --num-generated 200 \
  --output-csv outputs/qdiffusion_task_metrics.csv
```

Both evaluation scripts produce the same columns so the CSVs can be
compared directly:

| Metric | Direction | Description |
|--------|-----------|-------------|
| `Feature_FID` | ↓ lower better | FID in RetinaMNIST classifier feature space |
| `Feature_MMD_RBF` | ↓ lower better | Kernel MMD between real and fake features |
| `Fake_Target_Accuracy` | ↑ higher better | Fraction of fakes classified as intended class |
| `Fake_Target_Confidence` | ↑ higher better | Mean classifier confidence for intended class |
| `Fake_Prediction_MAE` | ↓ lower better | Ordinal distance: predicted class vs intended class |
| `Real_Target_Accuracy` | ↑ higher better | Classifier sanity check on real images |
| `Fake_To_Real_NN_Distance` | — | Mean nearest-neighbour distance fake→real (memorisation check) |
| `Fake_Diversity_Ratio` | ~1 ideal | Within-class NN distance: fake / real (mode-collapse check) |

---

## 8. Classical vs quantum comparison

Once both `outputs/ddpm_task_metrics.csv` and
`outputs/qdiffusion_task_metrics.csv` exist, run:

```bash
python scripts/compare_models.py \
  --ddpm-csv outputs/ddpm_task_metrics.csv \
  --qdiffusion-csv outputs/qdiffusion_task_metrics.csv \
  --output-dir outputs/comparison
```

Outputs:

```
outputs/comparison/
  combined_task_metrics.csv      ← stacked CSV with Model column (DDPM / QDiffusion)
  comparison_metrics.png         ← bar chart: each metric × class, DDPM vs QDiffusion
```

The comparison table printed to stdout shows per-class values and the
delta `QDiffusion − DDPM` for each metric.

---

## 9. Tables and plots for the paper

### FID/IS table from DDPM

```bash
python scripts/plot_metrics.py \
  --metrics-csv outputs/ddpm_metrics.csv \
  --output-dir reports/ddpm \
  --title "DDPM RetinaMNIST"
```

### Combined all-metrics table (DDPM only)

```bash
python scripts/combine_metric_csvs.py \
  --inputs outputs/ddpm_metrics.csv outputs/ddpm_task_metrics.csv \
  --output-csv outputs/ddpm_all_metrics.csv

python scripts/plot_metrics.py \
  --metrics-csv outputs/ddpm_all_metrics.csv \
  --output-dir reports/ddpm_all \
  --title "DDPM RetinaMNIST — all metrics"
```

### Classical vs quantum comparison figure

Use `outputs/comparison/comparison_metrics.png` (produced in step 8)
directly in the paper as the main comparison figure.

---

## 10. Statistical rigour (multi-seed runs)

To produce confidence intervals, run each model with at least 3 seeds and
aggregate:

```bash
for SEED in 42 7 123; do
  python scripts/train_qdiffusion_by_class.py \
    --config configs/qdiffusion_retinamnist.json \
    --seed $SEED \
    --output-root outputs/qdiffusion_seed${SEED}

  python scripts/evaluate_qdiffusion_task_metrics.py \
    --qdiffusion-root outputs/qdiffusion_seed${SEED} \
    --classifier-checkpoint outputs/classifier/retina_classifier.pt \
    --output-csv outputs/qdiffusion_task_metrics_seed${SEED}.csv \
    --seed $SEED
done
```

Then combine and compute mean ± std across seeds:

```bash
python scripts/combine_metric_csvs.py \
  --inputs \
    outputs/qdiffusion_task_metrics_seed42.csv \
    outputs/qdiffusion_task_metrics_seed7.csv \
    outputs/qdiffusion_task_metrics_seed123.csv \
  --output-csv outputs/qdiffusion_task_metrics_multiseed.csv
```

---

## Recommended evaluation cautions

- **FID/Inception Score are secondary for RetinaMNIST.** The Inception
  feature extractor is trained on natural images; use `Feature_FID` and
  `Feature_MMD_RBF` (computed in RetinaMNIST classifier feature space) as
  the primary distribution metrics.
- **Fake_Diversity_Ratio ≈ 1** is the target. Values far below 1 indicate
  mode collapse; values far above 1 may indicate class leakage or
  unrealistic dispersion.
- **Very low Fake_To_Real_NN_Distance** relative to
  `Real_Within_NN_Distance` is a memorisation signal.
- Central claims should rest on task-specific metrics and downstream
  augmentation utility, not FID alone.
