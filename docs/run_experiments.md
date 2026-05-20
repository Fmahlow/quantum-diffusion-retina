# Running experiments and generating tables

## 1. Environment

Use the same Python environment for all steps. The current notebook metadata says Python 3.11.11, but the shell currently resolves `python` to Anaconda Python 3.9 with a broken PyTorch install. Before running training, activate or create an environment where these imports work:

```bash
python -c "import torch, torchvision, diffusers, accelerate, medmnist, pandas, matplotlib; print(torch.__version__)"
```

Minimal dependencies:

```bash
pip install torch torchvision diffusers accelerate medmnist datasets pandas matplotlib
pip install "torchmetrics[image]" torch-fidelity
```

For the current DDPM training script, clone and install Hugging Face diffusers training examples:

```bash
git clone https://github.com/huggingface/diffusers.git
pip install -e "diffusers[training]"
pip install -r diffusers/examples/unconditional_image_generation/requirements.txt
```

If using Weights & Biases:

```bash
wandb login
```

Otherwise pass `--logger tensorboard`, or pass `--logger none` to omit the logger argument.

## 2. Export RetinaMNIST by class

```bash
python scripts/export_retinamnist.py \
  --split train \
  --output-dir data/retinamnist/train \
  --resize 32
```

This creates:

```text
data/retinamnist/train/class0
data/retinamnist/train/class1
data/retinamnist/train/class2
data/retinamnist/train/class3
data/retinamnist/train/class4
```

## 3. Train the classical DDPM baseline

First inspect the commands:

```bash
python scripts/train_ddpm_by_class.py --dry-run
```

Then run training:

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

For a quick smoke test, run only one class and fewer epochs:

```bash
python scripts/train_ddpm_by_class.py \
  --classes 0 \
  --epochs 1 \
  --logger tensorboard
```

## 4. Evaluate models

Evaluate against the official test split:

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

For CPU-only evaluation, add:

```bash
--no-fp16
```

## 5. Generate tables and plots

```bash
python scripts/plot_metrics.py \
  --metrics-csv outputs/ddpm_metrics.csv \
  --output-dir reports/ddpm \
  --title "DDPM RetinaMNIST"
```

Outputs:

```text
reports/ddpm/metrics_table.csv
reports/ddpm/metrics_table.md
reports/ddpm/metrics_table.tex
reports/ddpm/fid_by_class.png
reports/ddpm/inception_score_by_class.png
```

## 6. Task-specific evaluation

Train a RetinaMNIST classifier using only real training data. This classifier is not the generative model; it is an evaluation instrument for feature-space and downstream-domain metrics.

```bash
python scripts/train_retina_classifier.py \
  --output-dir outputs/classifier \
  --epochs 30 \
  --batch-size 64
```

Class weights are enabled by default to reduce class-imbalance bias in the evaluator. Use `--no-class-weights` only for ablation.

Then evaluate generated images with the trained classifier:

```bash
python scripts/evaluate_ddpm_task_metrics.py \
  --pipeline-root outputs/ddpm \
  --classifier-checkpoint outputs/classifier/retina_classifier.pt \
  --split test \
  --classes 0,1,2,3,4 \
  --num-generated 200 \
  --output-csv outputs/ddpm_task_metrics.csv
```

This produces:

```text
Feature_FID
Feature_MMD_RBF
Fake_Target_Accuracy
Fake_Target_Confidence
Fake_Prediction_MAE
Real_Target_Accuracy
Fake_To_Real_NN_Distance
Real_Within_NN_Distance
Fake_Within_NN_Distance
Fake_Diversity_Ratio
```

Generate the task-specific tables and plots:

```bash
python scripts/plot_metrics.py \
  --metrics-csv outputs/ddpm_task_metrics.csv \
  --output-dir reports/ddpm_task \
  --title "DDPM RetinaMNIST task-specific"
```

To create a single combined table with FID/IS and task-specific metrics:

```bash
python scripts/combine_metric_csvs.py \
  --inputs outputs/ddpm_metrics.csv outputs/ddpm_task_metrics.csv \
  --output-csv outputs/ddpm_all_metrics.csv

python scripts/plot_metrics.py \
  --metrics-csv outputs/ddpm_all_metrics.csv \
  --output-dir reports/ddpm_all \
  --title "DDPM RetinaMNIST all metrics"
```

## 7. Generate sample grids

```bash
python scripts/sample_ddpm_grid.py \
  --pipeline-root outputs/ddpm \
  --classes 0,1,2,3,4 \
  --num-samples 25 \
  --num-inference-steps 50 \
  --output-dir reports/ddpm/samples
```

For CPU-only sampling, add:

```bash
--no-fp16
```

Outputs:

```text
reports/ddpm/samples/class0_samples.png
reports/ddpm/samples/class1_samples.png
reports/ddpm/samples/class2_samples.png
reports/ddpm/samples/class3_samples.png
reports/ddpm/samples/class4_samples.png
```

## 8. Recommended paper figures

Use the DDPM metrics as baseline, but do not make FID/Inception Score the central scientific claim. For RetinaMNIST, the strongest evidence should be:

- class-wise generative metrics,
- generated-sample grids per class,
- nearest-neighbor checks against real train images,
- downstream classifier utility: real-only versus real+synthetic,
- confidence intervals across at least 3 seeds.

The quantum model should write its metrics in the same CSV schema as `outputs/ddpm_metrics.csv`; then `scripts/plot_metrics.py` can produce comparable tables and plots.
