"""Hybrid quantum-classical DDPM for RetinaMNIST synthesis.

Architecture:
  - Standard U-Net for 32x32 images with cosine noise schedule.
  - At the bottleneck (4x4 spatial resolution), a QuantumBottleneck module
    replaces the classical channel-attention block: it applies squeeze-and-
    excitation via a parameterized quantum circuit (PennyLane StronglyEntanglingLayers).
  - All classical layers run on whatever device (CPU/GPU) is requested.
  - The quantum layer uses PennyLane's `default.qubit` with `interface="torch"`,
    which follows the device of its input tensors (supports GPU via PyTorch ops).

Training:
  - One model per class, matching the DDPM-per-class protocol.
  - MSE loss on predicted noise (simplified DDPM objective, Ho et al. 2020).
  - Cosine beta schedule (Nichol & Dhariwal, Improved DDPM 2021).

Sampling:
  - Full DDPM reverse process (1000 steps) or fast DDIM-style (configurable).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule (Nichol & Dhariwal 2021)."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999).float()


class DDPMSchedule:
    """Precomputed DDPM schedule tensors."""

    def __init__(self, T: int = 1000, s: float = 0.008) -> None:
        self.T = T
        betas = cosine_beta_schedule(T, s)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)

        self.register("betas", betas)
        self.register("alphas", alphas)
        self.register("alphas_bar", alphas_bar)
        self.register("alphas_bar_prev", alphas_bar_prev)
        self.register("sqrt_alphas_bar", alphas_bar.sqrt())
        self.register("sqrt_one_minus_alphas_bar", (1.0 - alphas_bar).sqrt())
        # Posterior variance: σ²_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped", posterior_variance.clamp_min(1e-20).log())

    def register(self, name: str, value: torch.Tensor) -> None:
        setattr(self, name, value)

    def to(self, device: torch.device | str) -> "DDPMSchedule":
        for attr in [
            "betas", "alphas", "alphas_bar", "alphas_bar_prev",
            "sqrt_alphas_bar", "sqrt_one_minus_alphas_bar",
            "posterior_variance", "posterior_log_variance_clipped",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: q(x_t | x_0) = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicted mean and log-variance for the reverse step p(x_{t-1}|x_t)."""
        pred_noise = model(x_t, t)
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_1mab_t = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = (1.0 / self.alphas[t].sqrt()).view(-1, 1, 1, 1)
        mean = sqrt_recip_alpha_t * (x_t - betas_t / sqrt_1mab_t * pred_noise)
        log_var = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1)
        return mean, log_var

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """One reverse DDPM step."""
        batch_size = x_t.size(0)
        t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
        mean, log_var = self.p_mean_variance(model, x_t, t_tensor)
        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        return mean + (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        device: torch.device | str,
        stride: int = 1,
    ) -> torch.Tensor:
        """Full reverse diffusion sampling. stride > 1 for faster inference."""
        model.eval()
        x = torch.randn(shape, device=device)
        timesteps = list(range(0, self.T, stride))[::-1]
        for t in timesteps:
            x = self.p_sample(model, x, t)
        return x.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

def _sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal time step embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.cos(), args.sin()], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, t_dim: int = 256, emb_dim: int = 1024) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(t_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.t_dim = t_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = _sinusoidal_embedding(t, self.t_dim)
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ---------------------------------------------------------------------------
# Quantum bottleneck
# ---------------------------------------------------------------------------

def _build_quantum_layer(n_qubits: int, n_layers: int) -> qml.qnn.TorchLayer:
    """Build a PennyLane TorchLayer with StronglyEntanglingLayers ansatz."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
        # inputs: (n_qubits,) in [-π, π]  — angle embedding
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class QuantumBottleneck(nn.Module):
    """Squeeze-and-Excitation channel attention via parameterized quantum circuit.

    The module computes global average pooled channel features, maps them to
    n_qubits angles, processes them through a quantum circuit, then maps the
    expectation values back to per-channel sigmoid gates.  The spatial feature
    map is rescaled by these gates, analogously to a classical SE block.
    """

    def __init__(self, channels: int, n_qubits: int = 8, n_layers: int = 3) -> None:
        super().__init__()
        self.to_qubits = nn.Linear(channels, n_qubits)
        self.quantum_layer = _build_quantum_layer(n_qubits, n_layers)
        self.from_qubits = nn.Linear(n_qubits, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Global average pool → (B, C)
        gap = x.mean(dim=(-2, -1))
        # Map to angle range [-π, π]
        angles = torch.tanh(self.to_qubits(gap)) * math.pi
        # Quantum circuit → expectation values ∈ [-1, 1]^n_qubits
        q_out = self.quantum_layer(angles)
        # Back to channel scale gates ∈ (0, 1)
        gates = torch.sigmoid(self.from_qubits(q_out))
        return x * gates.view(B, C, 1, 1)


# ---------------------------------------------------------------------------
# Hybrid U-Net
# ---------------------------------------------------------------------------

class HybridUNet(nn.Module):
    """U-Net with a quantum bottleneck for 32x32 images.

    Down path:  3-ch input → 64 → 128 → 256 with 3× spatial downsampling.
    Bottleneck: ResBlock → QuantumBottleneck → ResBlock at 4×4 resolution.
    Up path:    256 → 128 → 64 → output with skip connections.
    """

    def __init__(
        self,
        img_channels: int = 3,
        base_ch: int = 64,
        t_dim: int = 256,
        emb_dim: int = 1024,
        n_qubits: int = 8,
        n_quantum_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        ch = base_ch
        self.time_emb = TimeEmbedding(t_dim=t_dim, emb_dim=emb_dim)

        # Encoder
        self.in_conv = nn.Conv2d(img_channels, ch, 3, padding=1)
        self.down1 = ResBlock(ch, ch, emb_dim, dropout)
        self.ds1 = Downsample(ch)          # 32→16

        self.down2 = ResBlock(ch, ch * 2, emb_dim, dropout)
        self.ds2 = Downsample(ch * 2)     # 16→8

        self.down3 = ResBlock(ch * 2, ch * 4, emb_dim, dropout)
        self.ds3 = Downsample(ch * 4)     # 8→4

        # Bottleneck (4×4 spatial)
        bot_ch = ch * 4
        self.bot1 = ResBlock(bot_ch, bot_ch, emb_dim, dropout)
        self.quantum_bot = QuantumBottleneck(bot_ch, n_qubits=n_qubits, n_layers=n_quantum_layers)
        self.bot2 = ResBlock(bot_ch, bot_ch, emb_dim, dropout)

        # Decoder (skip connections double the input channels)
        self.us3 = Upsample(bot_ch)
        self.up3 = ResBlock(bot_ch + ch * 4, ch * 2, emb_dim, dropout)

        self.us2 = Upsample(ch * 2)
        self.up2 = ResBlock(ch * 2 + ch * 2, ch, emb_dim, dropout)

        self.us1 = Upsample(ch)
        self.up1 = ResBlock(ch + ch, ch, emb_dim, dropout)

        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = nn.Conv2d(ch, img_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)

        # Encoder
        h0 = self.in_conv(x)
        h1 = self.down1(h0, t_emb)
        h1d = self.ds1(h1)

        h2 = self.down2(h1d, t_emb)
        h2d = self.ds2(h2)

        h3 = self.down3(h2d, t_emb)
        h3d = self.ds3(h3)

        # Bottleneck
        b = self.bot1(h3d, t_emb)
        b = self.quantum_bot(b)
        b = self.bot2(b, t_emb)

        # Decoder
        u3 = self.up3(torch.cat([self.us3(b), h3], dim=1), t_emb)
        u2 = self.up2(torch.cat([self.us2(u3), h2], dim=1), t_emb)
        u1 = self.up1(torch.cat([self.us1(u2), h1], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(u1)))


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class QDiffusionConfig:
    n_qubits: int = 8
    n_quantum_layers: int = 3
    img_channels: int = 3
    img_size: int = 32
    base_ch: int = 64
    t_dim: int = 256
    emb_dim: int = 1024
    dropout: float = 0.1
    T: int = 1000
    n_epochs: int = 300
    batch_size: int = 32
    lr: float = 2e-4
    grad_clip: float = 1.0
    seed: int = 42
    save_every: int = 100
    inference_stride: int = 1  # set > 1 for faster sampling during eval


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

@dataclass
class QDiffusionCheckpoint:
    config: QDiffusionConfig
    class_id: int
    epoch: int
    model_state: dict
    optimizer_state: dict
    losses: list[float] = field(default_factory=list)


def save_checkpoint(checkpoint: QDiffusionCheckpoint, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "config": checkpoint.config.__dict__,
        "class_id": checkpoint.class_id,
        "epoch": checkpoint.epoch,
        "model_state": checkpoint.model_state,
        "optimizer_state": checkpoint.optimizer_state,
        "losses": checkpoint.losses,
    }
    torch.save(data, path)


def load_checkpoint(path: Path, device: str = "cpu") -> QDiffusionCheckpoint:
    data = torch.load(path, map_location=device)
    cfg_dict = data["config"] if isinstance(data["config"], dict) else data["config"].__dict__
    config = QDiffusionConfig(**cfg_dict)
    return QDiffusionCheckpoint(
        config=config,
        class_id=data["class_id"],
        epoch=data["epoch"],
        model_state=data["model_state"],
        optimizer_state=data["optimizer_state"],
        losses=data.get("losses", []),
    )


def load_model_from_checkpoint(path: Path, device: str = "cpu") -> tuple[HybridUNet, QDiffusionConfig]:
    ckpt = load_checkpoint(path, device=device)
    cfg = ckpt.config
    model = HybridUNet(
        img_channels=cfg.img_channels,
        base_ch=cfg.base_ch,
        t_dim=cfg.t_dim,
        emb_dim=cfg.emb_dim,
        n_qubits=cfg.n_qubits,
        n_quantum_layers=cfg.n_quantum_layers,
        dropout=cfg.dropout,
    )
    model.load_state_dict(ckpt.model_state)
    model.to(device)
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_qdiffusion_for_class(
    real_images_01: torch.Tensor,
    class_id: int,
    output_dir: Path,
    config: QDiffusionConfig,
    device: str = "cuda",
    verbose: bool = True,
) -> QDiffusionCheckpoint:
    """Train one hybrid quantum DDPM for a single class.

    Args:
        real_images_01: float32 tensor (N, C, H, W) in [0, 1].
        class_id: integer class label for naming checkpoints.
        output_dir: directory to save checkpoints and final model.
        config: training hyperparameters.
        device: torch device string.
        verbose: print loss every epoch.

    Returns:
        Final QDiffusionCheckpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map [0,1] to [-1,1] for diffusion
    real_images = real_images_01 * 2.0 - 1.0
    dataset = torch.utils.data.TensorDataset(real_images)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    schedule = DDPMSchedule(T=config.T).to(device)

    model = HybridUNet(
        img_channels=config.img_channels,
        base_ch=config.base_ch,
        t_dim=config.t_dim,
        emb_dim=config.emb_dim,
        n_qubits=config.n_qubits,
        n_quantum_layers=config.n_quantum_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    losses: list[float] = []

    for epoch in range(1, config.n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (x0_batch,) in loader:
            x0_batch = x0_batch.to(device)
            n = x0_batch.size(0)

            t = torch.randint(0, config.T, (n,), device=device, dtype=torch.long)
            x_t, noise = schedule.q_sample(x0_batch, t)

            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if verbose:
            print(
                f"[class {class_id}] epoch {epoch:4d}/{config.n_epochs}  loss={avg_loss:.6f}",
                flush=True,
            )

        if epoch % config.save_every == 0 or epoch == config.n_epochs:
            ckpt = QDiffusionCheckpoint(
                config=config,
                class_id=class_id,
                epoch=epoch,
                model_state={k: v.cpu() for k, v in model.state_dict().items()},
                optimizer_state=optimizer.state_dict(),
                losses=losses,
            )
            save_checkpoint(ckpt, output_dir / f"checkpoint_epoch{epoch:04d}.pt")

    final_ckpt = QDiffusionCheckpoint(
        config=config,
        class_id=class_id,
        epoch=config.n_epochs,
        model_state={k: v.cpu() for k, v in model.state_dict().items()},
        optimizer_state=optimizer.state_dict(),
        losses=losses,
    )
    save_checkpoint(final_ckpt, output_dir / "final.pt")
    return final_ckpt


# ---------------------------------------------------------------------------
# Sampling utility
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_qdiffusion(
    model: HybridUNet,
    config: QDiffusionConfig,
    n_samples: int,
    device: str,
    seed: int = 0,
) -> torch.Tensor:
    """Generate n_samples images in [0, 1] from a trained HybridUNet.

    Returns:
        Tensor (n_samples, img_channels, img_size, img_size) in [0, 1].
    """
    torch.manual_seed(seed)
    schedule = DDPMSchedule(T=config.T).to(device)
    model.eval()
    model.to(device)

    all_images: list[torch.Tensor] = []
    remaining = n_samples
    batch_size = min(16, n_samples)

    while remaining > 0:
        n = min(batch_size, remaining)
        shape = (n, config.img_channels, config.img_size, config.img_size)
        images = schedule.sample(model, shape, device=device, stride=config.inference_stride)
        # Map from [-1, 1] to [0, 1]
        images_01 = (images + 1.0) * 0.5
        all_images.append(images_01.cpu())
        remaining -= n

    return torch.cat(all_images, dim=0)
