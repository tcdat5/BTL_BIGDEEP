"""4 model DL dạng MLP cho dữ liệu bảng: AE, VAE, DAE, DNN classifier (PROJECT_SPEC 5.3).

Linear -> BatchNorm1d -> ReLU cho mỗi tầng ẩn; tầng latent (bottleneck) và tầng ra là Linear thuần.
Nhiễu của DAE nằm trong forward() và chỉ bật khi training -> graph ONNX (eval) không có nhiễu.
"""
from __future__ import annotations

import torch
from torch import nn

MODEL_SPECS = {
    "ae": {"name": "DeepAutoencoder", "group": "unsupervised"},
    "vae": {"name": "VariationalAE", "group": "unsupervised"},
    "dae": {"name": "DenoisingAE", "group": "unsupervised"},
    "mlp": {"name": "DNN_Classifier", "group": "supervised"},
}


def mlp_stack(dims, dropout: float = 0.0) -> list:
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
    return layers


def make_decoder(d: int, hidden, latent: int) -> nn.Sequential:
    rev = list(reversed(hidden))
    return nn.Sequential(*mlp_stack([latent] + rev), nn.Linear(rev[-1], d))


class AE(nn.Module):
    def __init__(self, d: int, hidden=(64, 32), latent: int = 16):
        super().__init__()
        hidden = list(hidden)
        self.encoder = nn.Sequential(*mlp_stack([d] + hidden), nn.Linear(hidden[-1], latent))
        self.decoder = make_decoder(d, hidden, latent)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DAE(AE):
    def __init__(self, d: int, hidden=(64, 32), latent: int = 16, noise_std: float = 0.1):
        super().__init__(d, hidden, latent)
        self.noise_std = noise_std

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return self.decoder(self.encoder(x))


class VAE(nn.Module):
    def __init__(self, d: int, hidden=(64, 32), latent: int = 16):
        super().__init__()
        hidden = list(hidden)
        self.body = nn.Sequential(*mlp_stack([d] + hidden))
        self.mu = nn.Linear(hidden[-1], latent)
        self.logvar = nn.Linear(hidden[-1], latent)
        self.decoder = make_decoder(d, hidden, latent)

    def encode(self, x):
        h = self.body(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.decoder(z), mu, logvar

    def reconstruct_mean(self, x):
        mu, _ = self.encode(x)
        return self.decoder(mu)


def vae_loss(x, recon, mu, logvar, beta: float):
    """MSE tổng theo feature, trung bình theo batch + beta * KL."""
    rec = ((recon - x) ** 2).sum(dim=1).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    return rec + beta * kl, rec, kl


class MLPClassifier(nn.Module):
    def __init__(self, d: int, hidden=(128, 64, 32), dropout: float = 0.2):
        super().__init__()
        hidden = list(hidden)
        self.net = nn.Sequential(*mlp_stack([d] + hidden, dropout), nn.Linear(hidden[-1], 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ReconScore(nn.Module):
    """Score = MSE tái tạo theo mẫu (VAE dùng μ). Đây là graph được export ONNX."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        recon = self.model.reconstruct_mean(x) if isinstance(self.model, VAE) else self.model(x)
        return ((recon - x) ** 2).mean(dim=1)


class ProbScore(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x))


def build_model(kind: str, d: int, hidden=(64, 32), latent: int = 16, noise_std: float = 0.1) -> nn.Module:
    if kind == "ae":
        return AE(d, hidden, latent)
    if kind == "dae":
        return DAE(d, hidden, latent, noise_std)
    if kind == "vae":
        return VAE(d, hidden, latent)
    if kind == "mlp":
        return MLPClassifier(d)
    raise ValueError("unknown model kind: %s" % kind)


def score_module(kind: str, model: nn.Module) -> nn.Module:
    return ProbScore(model) if kind == "mlp" else ReconScore(model)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def encode_latent(model: nn.Module, x):
    """Vector latent cho EDA: AE/DAE -> output encoder; VAE -> μ. (MLP không có latent.)"""
    if isinstance(model, VAE):
        return model.encode(x)[0]
    if isinstance(model, AE):
        return model.encoder(x)
    raise ValueError("model không có latent: %s" % type(model).__name__)


def save_checkpoint(path, kind: str, model: nn.Module, d: int, hidden, latent: int, noise_std: float) -> None:
    torch.save({"kind": kind, "d": d, "hidden": list(hidden), "latent": latent, "noise_std": noise_std,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, str(path))


def load_checkpoint(path):
    """-> (kind, model ở chế độ eval trên CPU)."""
    ck = torch.load(str(path), map_location="cpu")
    model = build_model(ck["kind"], ck["d"], hidden=tuple(ck["hidden"]), latent=ck["latent"],
                        noise_std=ck["noise_std"])
    model.load_state_dict(ck["state_dict"])
    return ck["kind"], model.eval()
