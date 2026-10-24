"""Stage 5: frozen MiniLM encoder (h0) -> 3-hidden-layer MLP (h1, h2, h3) -> shortlist score."""

import numpy as np
import torch
from torch import nn

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_encoder = None


def encode(texts: list[str]) -> np.ndarray:
    """h0: 384-d sentence embeddings from the frozen encoder."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(ENCODER_NAME)
    return _encoder.encode(list(texts), batch_size=32, convert_to_numpy=True)


class HiringMLP(nn.Module):
    def __init__(self, d_in: int = 384, hidden: tuple[int, ...] = (256, 128, 64)):
        super().__init__()
        dims = (d_in, *hidden)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.head = nn.Linear(dims[-1], 1)

    def forward(self, h0: torch.Tensor, return_hidden: bool = False):
        hidden = [h0]
        h = h0
        for layer in self.layers:
            h = torch.relu(layer(h))
            hidden.append(h)
        logit = self.head(h).squeeze(-1)
        return (logit, hidden) if return_hidden else logit


def train_mlp(x: np.ndarray, y: np.ndarray, seed: int, epochs: int = 300, lr: float = 1e-3) -> HiringMLP:
    """Full-batch training. The same seed gives M0 and M_b identical initial weights."""
    torch.manual_seed(seed)
    model = HiringMLP(d_in=x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def run(model: HiringMLP, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Returns shortlist scores in [0, 1] and hidden states [h0, h1, h2, h3]."""
    logit, hidden = model(torch.as_tensor(x, dtype=torch.float32), return_hidden=True)
    return torch.sigmoid(logit).numpy(), [h.numpy() for h in hidden]
