"""Choose the shared training config by 5-fold cross-validation on the training resumes.

The test pairs are never used. Every model in run_experiment.py then uses the winning config,
so models still differ only in their labels or mitigation.

    python tune_regularisation.py --seeds 5
"""

import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from biasaudit import data, model

GRID = {
    "hidden": [(256, 128, 64), (64, 32, 16)],
    "dropout": [0.0, 0.3, 0.5],
    "weight_decay": [1e-4, 1e-2],
    "epochs": [100, 300],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    raw = pd.read_csv(data.RESUME_CSV)
    configs = [model.TrainConfig(**dict(zip(GRID, values))) for values in itertools.product(*GRID.values())]
    rows = []
    for seed in range(args.seeds):
        exp = data.make_experiment(seed, df=raw)
        x, y = model.encode(exp.train.text), exp.train.y_true.to_numpy()  # Set A
        folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y))
        for cfg in configs:
            oof, train_acc = np.zeros(len(y)), []
            for k, (tr, va) in enumerate(folds):
                m = model.train_mlp(x[tr], y[tr], seed + k, cfg)
                oof[va], _ = model.run(m, x[va])
                s_tr, _ = model.run(m, x[tr])
                train_acc.append(np.mean((s_tr >= 0.5) == y[tr]))
            rows.append({
                "hidden": "x".join(map(str, cfg.hidden)), "dropout": cfg.dropout,
                "weight_decay": cfg.weight_decay, "epochs": cfg.epochs,
                "cv_log_loss": log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6)),
                "cv_acc": np.mean((oof >= 0.5) == y), "train_acc": np.mean(train_acc),
            })
        print(f"seed {seed} done")

    df = pd.DataFrame(rows)
    keys = ["hidden", "dropout", "weight_decay", "epochs"]
    summary = df.groupby(keys).agg(
        cv_log_loss=("cv_log_loss", "mean"), cv_log_loss_sd=("cv_log_loss", "std"),
        cv_acc=("cv_acc", "mean"), train_acc=("train_acc", "mean"),
    )
    summary["overfit_gap"] = summary.train_acc - summary.cv_acc
    summary = summary.sort_values("cv_log_loss")
    summary.to_csv("tuning_results.csv")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
