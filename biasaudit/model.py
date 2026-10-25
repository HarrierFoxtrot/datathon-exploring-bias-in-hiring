"""Stage 5: frozen MiniLM encoder (h0) -> 3-hidden-layer MLP (h1, h2, h3) -> shortlist score."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_encoder = None


@dataclass(frozen=True)
class TrainConfig:
    """Shared by every model, so models differ only in their labels or mitigation.

    The default was chosen by 5-fold cross-validation on the training resumes
    (tune_regularisation.py): CV log-loss 0.159 vs 0.278 for UNREGULARISED,
    which reproduces the first results.
    """
    hidden: tuple[int, ...] = (64, 32, 16)
    dropout: float = 0.3
    weight_decay: float = 1e-2
    epochs: int = 300
    lr: float = 1e-3


REGULARISED = TrainConfig()
UNREGULARISED = TrainConfig(hidden=(256, 128, 64), dropout=0.0, weight_decay=1e-4)


def encode(texts: list[str]) -> np.ndarray:
    """h0: 384-d sentence embeddings from the frozen encoder."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(ENCODER_NAME)
    return _encoder.encode(list(texts), batch_size=32, convert_to_numpy=True)


class HiringMLP(nn.Module):
    def __init__(self, d_in: int = 384, hidden: tuple[int, ...] = (256, 128, 64), dropout: float = 0.0):
        super().__init__()
        dims = (d_in, *hidden)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.head = nn.Linear(dims[-1], 1)
        self.dropout = nn.Dropout(dropout)  # identity in eval mode, so audits see the full network

    def forward(self, h0: torch.Tensor, return_hidden: bool = False):
        hidden = [h0]
        h = h0
        for layer in self.layers:
            h = torch.relu(layer(h))
            hidden.append(h)
            h = self.dropout(h)
        logit = self.head(h).squeeze(-1)
        return (logit, hidden) if return_hidden else logit


def build_mlp(d_in: int, seed: int, cfg: TrainConfig) -> HiringMLP:
    """The same seed gives every model identical initial weights."""
    torch.manual_seed(seed)
    return HiringMLP(d_in=d_in, hidden=cfg.hidden, dropout=cfg.dropout)


def train_mlp(x: np.ndarray, y: np.ndarray, seed: int, cfg: TrainConfig = REGULARISED) -> HiringMLP:
    """Full-batch training with binary cross-entropy."""
    model = build_mlp(x.shape[1], seed, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    model.train()
    for _ in range(cfg.epochs):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def run(model: nn.Module, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Returns shortlist scores in [0, 1] and hidden states [h0, h1, h2, h3]."""
    logit, hidden = model(torch.as_tensor(x, dtype=torch.float32), return_hidden=True)
    return torch.sigmoid(logit).numpy(), [h.numpy() for h in hidden]
