"""Part 1, stages 1-8: build the data, train M0 and M_b, mitigate M_b, and audit every model.

    python run_experiment.py --seeds 5
    python run_experiment.py --seeds 5 --lam 2 --out results_lam2
"""

import argparse

import numpy as np
import pandas as pd

from biasaudit import audit, data, mitigate, model

MODEL_ORDER = ["M0", "M_b", "M_adv", "M_proj", "M_proj_out"]


def run_seed(seed: int, flip_p: float, lam: float, raw: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    exp = data.make_experiment(seed, flip_p=flip_p, df=raw)
    x_train = model.encode(exp.train.text)
    x_train_f, x_train_m = model.encode(exp.train.text_f), model.encode(exp.train.text_m)
    x_f, x_m = model.encode(exp.test.text_f), model.encode(exp.test.text_m)
    y_test = exp.test.y_true.to_numpy()
    y_biased = exp.train.y_biased.to_numpy()

    m_b = model.train_mlp(x_train, y_biased, seed=seed)
    models = {
        "M0": model.train_mlp(x_train, exp.train.y_true.to_numpy(), seed=seed),
        "M_b": m_b,
        # Stage 7(a): retrain on the biased labels with gender adversaries on h1-h3.
        "M_adv": mitigate.train_adversarial(x_train, y_biased, (exp.train.gender == "F").to_numpy(), seed, lam=lam),
        # Stage 7(b): project h0, where the name enters. v_l comes from training pairs, so test stays unseen.
        # After this, h1-h3 probes are already at chance; projecting them as well only costs accuracy,
        # because at late layers the gender direction is nearly the hiring direction (cos ~ -0.95).
        "M_proj": mitigate.fit_projection(m_b, x_train_f, x_train_m, layers=(0,)),
        # Ablation: project only the last layer, an output-only fix.
        "M_proj_out": mitigate.fit_projection(m_b, x_train_f, x_train_m, layers=(3,)),
    }

    # Stage 8: the same audit on every model.
    behav, act = [], []
    for name, m in models.items():
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
    parser.add_argument("--lam", type=float, default=1.0, help="gradient reversal strength for M_adv")
    parser.add_argument("--out", default="results", help="prefix for the output CSVs")
    args = parser.parse_args()

    raw = pd.read_csv(data.RESUME_CSV)
    behav, act = [], []
    for seed in range(args.seeds):
        b, a = run_seed(seed, args.flip_p, args.lam, raw)
        behav += b
        act += a

    behav, act = pd.DataFrame(behav), pd.DataFrame(act)
    for df in (behav, act):
        df["model"] = pd.Categorical(df.model, categories=MODEL_ORDER, ordered=True)
    behav.to_csv(f"{args.out}_behavioural.csv", index=False)
    act.to_csv(f"{args.out}_activation.csv", index=False)

    agg = lambda s: f"{s.mean():+.3f} ± {s.std():.3f}"
    print("\nBehavioural bias (mean ± sd over seeds)")
    print(behav.drop(columns="seed").groupby("model", observed=True).agg(agg).T.to_string())
    for metric in ("probe_acc", "v_ratio"):
        print(f"\n{metric} per layer")
        table = act.groupby(["layer", "model"], observed=True)[metric].agg(agg).unstack("model")
        print(table.to_string())


if __name__ == "__main__":
    main()
