"""Part 1, stages 1-6: build the data, train M0 and M_b, and measure their bias.

    python run_experiment.py --seeds 5
"""

import argparse

import numpy as np
import pandas as pd

from biasaudit import audit, data, model


def run_seed(seed: int, flip_p: float, raw: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    exp = data.make_experiment(seed, flip_p=flip_p, df=raw)
    x_train = model.encode(exp.train.text)
    x_f, x_m = model.encode(exp.test.text_f), model.encode(exp.test.text_m)
    y_test = exp.test.y_true.to_numpy()

    behav, act = [], []
    for name, labels in [("M0", exp.train.y_true), ("M_b", exp.train.y_biased)]:
        m = model.train_mlp(x_train, labels.to_numpy(), seed=seed)
        s_f, hid_f = model.run(m, x_f)
        s_m, hid_m = model.run(m, x_m)
        behav.append({"seed": seed, "model": name, **audit.behavioural(s_f, s_m, y_test)})
        act += [{"seed": seed, "model": name, **row} for row in audit.activation(hid_f, hid_m, seed)]

    n_flipped = int((exp.train.y_true != exp.train.y_biased).sum())
    print(f"seed {seed}: train={len(exp.train)} test pairs={len(exp.test)} labels flipped in Set B={n_flipped}")
    return behav, act


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--flip-p", type=float, default=0.3)
    args = parser.parse_args()

    raw = pd.read_csv(data.RESUME_CSV)
    behav, act = [], []
    for seed in range(args.seeds):
        b, a = run_seed(seed, args.flip_p, raw)
        behav += b
        act += a

    behav, act = pd.DataFrame(behav), pd.DataFrame(act)
    behav.to_csv("results_behavioural.csv", index=False)
    act.to_csv("results_activation.csv", index=False)

    agg = lambda s: f"{s.mean():+.3f} ± {s.std():.3f}"
    print("\nBehavioural bias (mean ± sd over seeds)")
    print(behav.drop(columns="seed").groupby("model").agg(agg).T.to_string())
    print("\nActivation-space bias per layer")
    print(act.drop(columns="seed").groupby(["layer", "model"]).agg(agg).unstack("model").to_string())


if __name__ == "__main__":
    main()
