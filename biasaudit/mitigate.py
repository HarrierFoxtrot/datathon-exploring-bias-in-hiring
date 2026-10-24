"""Stage 7: remove bias from M_b by (a) adversarial training or (b) orthogonal projection.

Both return a module with the same forward(h0, return_hidden) interface as HiringMLP,
so model.run and the stage 6 audit work on them unchanged.
"""

import numpy as np
import torch
from torch import nn

from .model import HiringMLP


# (b) Orthogonal projection -------------------------------------------------

class ProjectedMLP(nn.Module):
    """A trained HiringMLP with h_l replaced by mu_l + P_l (h_l - mu_l) at the chosen layers.

    Centring on mu_l removes only the gender *variation* along v_l. Projecting through the
    origin would also delete the mean value along v_l, shifting every score by a constant.
    """

    def __init__(self, base: HiringMLP):
        super().__init__()
        self.base = base
        self.projections: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _project(self, l: int, h: torch.Tensor) -> torch.Tensor:
        if l not in self.projections:
            return h
        P, mu = self.projections[l]
        return (h - mu) @ P + mu  # P is symmetric, so x @ P == (P x^T)^T

    def forward(self, h0: torch.Tensor, return_hidden: bool = False):
        h = self._project(0, h0)
        hidden = [h]
        for l, layer in enumerate(self.base.layers, start=1):
            h = self._project(l, torch.relu(layer(h)))
            hidden.append(h)
        logit = self.base.head(h).squeeze(-1)
        return (logit, hidden) if return_hidden else logit


def projection_matrix(v: np.ndarray) -> torch.Tensor:
    """P = I - v v^T / v^T v: removes the direction v and keeps everything orthogonal to it."""
    v = torch.as_tensor(v, dtype=torch.float32)
    return torch.eye(len(v)) - torch.outer(v, v) / (v @ v)


@torch.no_grad()
def fit_projection(base: HiringMLP, x_f: np.ndarray, x_m: np.ndarray, layers=(0, 1, 2, 3)) -> ProjectedMLP:
    """Estimate v_l from counterfactual pairs and project, one layer at a time.

    Each v_l is measured *after* the earlier projections, because later layers can rebuild the bias.
    """
    model = ProjectedMLP(base)
    xf = torch.as_tensor(x_f, dtype=torch.float32)
    xm = torch.as_tensor(x_m, dtype=torch.float32)
    for l in sorted(layers):
        _, hid_f = model(xf, return_hidden=True)
        _, hid_m = model(xm, return_hidden=True)
        v = (hid_f[l] - hid_m[l]).mean(dim=0).numpy()
        mu = torch.cat([hid_f[l], hid_m[l]]).mean(dim=0)
        model.projections[l] = (projection_matrix(v), mu)
    return model.eval()


# (a) Adversarial debiasing ---------------------------------------------------

class GradReverse(torch.autograd.Function):
    """Identity going forward; multiplies the gradient by -lam going back."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def train_adversarial(
    x: np.ndarray, y: np.ndarray, is_female: np.ndarray, seed: int,
    lam: float = 1.0, layers=(1, 2, 3), epochs: int = 300, lr: float = 1e-3,
) -> HiringMLP:
    """Train on the biased labels while gender adversaries on h1-h3 try to recover gender.

    The adversaries minimise their loss; the gradient reversal makes the MLP layers maximise it.
    h0 is the frozen encoder, so there is nothing to train there.
    """
    torch.manual_seed(seed)
    model = HiringMLP(d_in=x.shape[1])  # built first, so it starts from the same weights as M0 and M_b
    dims = [x.shape[1]] + [layer.out_features for layer in model.layers]
    adversaries = nn.ModuleList(
        nn.Sequential(nn.Linear(dims[l], 32), nn.ReLU(), nn.Linear(32, 1)) for l in layers
    )
    opt = torch.optim.Adam([*model.parameters(), *adversaries.parameters()], lr=lr, weight_decay=1e-4)
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    gt = torch.as_tensor(is_female, dtype=torch.float32)
    bce = nn.functional.binary_cross_entropy_with_logits

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logit, hidden = model(xt, return_hidden=True)
        loss = bce(logit, yt)
        for l, adv in zip(layers, adversaries):
            loss = loss + bce(adv(GradReverse.apply(hidden[l], lam)).squeeze(-1), gt)
        loss.backward()
        opt.step()
    model.eval()
    return model
