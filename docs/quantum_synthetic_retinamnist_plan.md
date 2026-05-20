# Scientific plan: quantum synthetic RetinaMNIST

## Starting point

The current notebook trains one class-specific unconditional DDPM for RetinaMNIST. It is therefore a diffusion baseline, not a GAN baseline. RetinaMNIST has 1,600 fundus images with the official 1,080 / 120 / 400 train / validation / test split and an ordinal 5-class target, so the evaluation must preserve the official split and report class-wise behavior.

## Research question

Can a quantum or hybrid quantum-classical generative model produce RetinaMNIST-like synthetic data that is useful for data augmentation, while remaining competitive with a classical generative baseline under controlled compute, data, and evaluation protocols?

## Proposed quantum model

The first rigorous model should be a hybrid quantum-classical GAN:

- Generator: a parameterized quantum circuit sampled through expectation values or measurement probabilities, followed by a small classical decoder to map the quantum latent representation to a 28x28 or 32x32 image.
- Discriminator: a classical CNN discriminator, initially WGAN-GP for training stability.
- Conditioning: either one generator per class, matching the current DDPM protocol, or class conditioning through a label embedding if compute allows.
- Framework: PennyLane + PyTorch, because it supports differentiable quantum nodes inside PyTorch training loops.

The first ablation should be a Quantum Circuit Born Machine (QCBM) or QCBM-latent model. In this variant, the circuit learns a latent distribution and a fixed or trained classical decoder maps latent samples to images. This is more defensible on simulators because the number of qubits can remain small.

## Experimental protocol

1. Data audit: export official train/val/test splits, count labels, visualize samples, and freeze preprocessing.
2. Classical baselines: keep the DDPM baseline and add a lightweight DCGAN or WGAN-GP if the paper needs a direct GAN comparison.
3. Quantum model: train at least one quantum/hybrid generator with fixed qubit count, ansatz depth, shots/simulator settings, optimizer, and seeds.
4. Ablations: vary qubits, circuit depth, latent dimension, and per-class versus conditional training.
5. Evaluation: report class-wise FID/IS for comparability, RetinaMNIST-classifier Feature FID, feature-space MMD, target-class fidelity, ordinal prediction MAE, nearest-neighbor diversity/memorization checks, and downstream utility with train-on-synthetic-test-on-real plus augmentation experiments.
6. Statistical rigor: run at least 3 seeds, bootstrap confidence intervals for metrics, and report failure cases and mode collapse checks.
7. Reproducibility: version dependencies, save configs, random seeds, generated samples, checkpoints, and metric CSVs.

## Evaluation cautions

FID and Inception Score are weak for small biomedical images because the Inception feature extractor is trained on natural images. They can remain for comparability, but the main claims should rely on task-specific downstream performance, RetinaMNIST classifier feature metrics, target-class fidelity, and explicit memorization/diversity checks.

## Code structure

The project should keep notebooks as experiment narratives and move reusable code into:

- `src/retina_synthesis/data.py`: dataset loading, label names, class export, label counts.
- `src/retina_synthesis/image_utils.py`: image and tensor conversion helpers.
- `src/retina_synthesis/metrics.py`: reusable evaluation logic.
- `src/retina_synthesis/generators/diffusion.py`: DDPM command construction.
- `src/retina_synthesis/quantum/`: quantum generator implementations.
- `scripts/`: reproducible command-line entry points.
- `configs/`: frozen experiment parameters.

## Initial references

- MedMNIST v2 / RetinaMNIST: https://medmnist.com/
- MedMNIST v2 paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC9852451/
- Quantum GANs: https://arxiv.org/abs/1804.08641
- Quantum Circuit Born Machine: https://www.nature.com/articles/s41534-019-0157-8
- MosaiQ image-generation QGAN: https://openaccess.thecvf.com/content/ICCV2023/html/Silver_MosaiQ_Quantum_Generative_Adversarial_Networks_for_Image_Generation_on_NISQ_ICCV_2023_paper.html
- PennyLane PyTorch interface: https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html
