"""Stage 6: behavioural and activation-space bias on counterfactual test pairs."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def behavioural(s_f: np.ndarray, s_m: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    d_f, d_m = s_f >= threshold, s_m >= threshold
    pos = y == 1
    return {
        "score_gap": float(np.mean(s_f - s_m)),             # < 0: women scored lower
        "flip_rate": float(np.mean(d_f != d_m)),
        "tpr_f": float(d_f[pos].mean()),
        "tpr_m": float(d_m[pos].mean()),
        "tpr_gap": float(d_f[pos].mean() - d_m[pos].mean()),  # equal opportunity
        "accuracy": float(np.mean(np.concatenate([d_f == y, d_m == y]))),
    }


def bias_vector_ratio(h_f: np.ndarray, h_m: np.ndarray) -> float:
    """||v_l|| / E||h_l||, with v_l the mean female-minus-male shift."""
    v = (h_f - h_m).mean(axis=0)
    scale = np.linalg.norm(np.concatenate([h_f, h_m]), axis=1).mean()
    return float(np.linalg.norm(v) / scale) if scale > 0 else 0.0


def probe_accuracy(h_f: np.ndarray, h_m: np.ndarray, seed: int, n_splits: int = 20) -> float:
    """Held-out accuracy of a linear gender probe.

    Splits by resume, so both versions of a resume sit on the same side and the
    probe cannot win by memorising resume content. Averaged over random splits.
    """
    rng = np.random.default_rng(seed)
    n = len(h_f)
    accs = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        tr, te = perm[: n // 2], perm[n // 2:]
        x_tr = np.concatenate([h_f[tr], h_m[tr]])
        x_te = np.concatenate([h_f[te], h_m[te]])
        g_tr = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
        g_te = np.r_[np.ones(len(te)), np.zeros(len(te))]
        if x_tr.std(axis=0).max() == 0:  # dead layer: nothing to probe
            accs.append(0.5)
            continue
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        accs.append(probe.fit(x_tr, g_tr).score(x_te, g_te))
    return float(np.mean(accs))


def activation(hid_f: list[np.ndarray], hid_m: list[np.ndarray], seed: int) -> list[dict]:
    return [
        {"layer": f"h{l}", "v_ratio": bias_vector_ratio(hf, hm), "probe_acc": probe_accuracy(hf, hm, seed)}
        for l, (hf, hm) in enumerate(zip(hid_f, hid_m))
    ]
