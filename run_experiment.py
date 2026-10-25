"""Part 1, stages 1-8: build the data, train the models, mitigate, and audit every model.

Two bias experiments share the same resumes, test pairs and control M0:
  label bias           (Set B): female engineers' labels flipped to 0 with p = flip_p
  representation bias  (Set C): every label correct, but `skew` of engineers are men
                                and `skew` of non-engineers are women

    python run_experiment.py --seeds 5
    python run_experiment.py --seeds 5 --config unregularised --out results_unreg
"""

import argparse

import numpy as np
import pandas as pd

from biasaudit import audit, data, mitigate, model

EXPERIMENTS = {
    "label bias": ["M0", "M_b", "M_adv", "M_proj", "M_proj_out"],
    "representation bias": ["M0", "M_c", "M_c_adv", "M_c_proj"],
}
MODEL_ORDER = list(dict.fromkeys(m for ms in EXPERIMENTS.values() for m in ms))
CONFIGS = {"regularised": model.REGULARISED, "unregularised": model.UNREGULARISED}


def run_seed(seed: int, args, cfg: model.TrainConfig, raw: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    exp = data.make_experiment(seed, flip_p=args.flip_p, skew=args.skew, df=raw)
    tr = exp.train
    x_train, x_train_c = model.encode(tr.text), model.encode(tr.text_c)
    x_train_f, x_train_m = model.encode(tr.text_f), model.encode(tr.text_m)
    x_f, x_m = model.encode(exp.test.text_f), model.encode(exp.test.text_m)
    y_test, y_true, y_biased = exp.test.y_true.to_numpy(), tr.y_true.to_numpy(), tr.y_biased.to_numpy()

    m_b = model.train_mlp(x_train, y_biased, seed, cfg)
    m_c = model.train_mlp(x_train_c, y_true, seed, cfg)
    models = {
        "M0": model.train_mlp(x_train, y_true, seed, cfg),
        # Label bias. Stage 7(a): adversaries on h1-h3; 7(b): project h0, where the name enters.
        # v_l comes from training pairs, so the test set stays unseen.
        "M_b": m_b,
        "M_adv": mitigate.train_adversarial(x_train, y_biased, (tr.gender == "F").to_numpy(), seed, args.lam, cfg=cfg),
        "M_proj": mitigate.fit_projection(m_b, x_train_f, x_train_m, layers=(0,)),
        # Ablation: project only the last layer, an output-only fix.
        "M_proj_out": mitigate.fit_projection(m_b, x_train_f, x_train_m, layers=(3,)),
        # Representation bias: same mitigations applied to M_c.
        "M_c": m_c,
        "M_c_adv": mitigate.train_adversarial(x_train_c, y_true, (tr.gender_c == "F").to_numpy(), seed, args.lam, cfg=cfg),
        "M_c_proj": mitigate.fit_projection(m_c, x_train_f, x_train_m, layers=(0,)),
    }

    # Stage 8: the same audit on every model.
    behav, act = [], []
    for name, m in models.items():
        s_f, hid_f = model.run(m, x_f)
        s_m, hid_m = model.run(m, x_m)
        behav.append({"seed": seed, "model": name, **audit.behavioural(s_f, s_m, y_test)})
        act += [{"seed": seed, "model": name, **row} for row in audit.activation(hid_f, hid_m, seed)]

    n_flipped = int((y_true != y_biased).sum())
    men_eng = (tr.gender_c[tr.y_true == 1] == "M").mean()
    print(f"seed {seed}: train={len(tr)} test pairs={len(exp.test)} | Set B flipped={n_flipped} | "
          f"Set C engineers male={men_eng:.0%}")
    return behav, act


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--flip-p", type=float, default=0.3, help="Set B: chance a female engineer's label is flipped")
    parser.add_argument("--skew", type=float, default=0.8, help="Set C: share of engineers who are men (and non-engineers who are women)")
    parser.add_argument("--lam", type=float, default=1.0, help="gradient reversal strength for the adversarial models")
    parser.add_argument("--config", choices=CONFIGS, default="regularised")
    parser.add_argument("--out", default="results", help="prefix for the output CSVs")
    args = parser.parse_args()
    cfg = CONFIGS[args.config]
    print(f"config: {cfg}")

    raw = pd.read_csv(data.RESUME_CSV)
    behav, act = [], []
    for seed in range(args.seeds):
        b, a = run_seed(seed, args, cfg, raw)
        behav += b
        act += a

    behav, act = pd.DataFrame(behav), pd.DataFrame(act)
    for df in (behav, act):
        df["model"] = pd.Categorical(df.model, categories=MODEL_ORDER, ordered=True)
    behav.to_csv(f"{args.out}_behavioural.csv", index=False)
    act.to_csv(f"{args.out}_activation.csv", index=False)

    agg = lambda s: f"{s.mean():+.3f} ± {s.std():.3f}"
    for experiment, names in EXPERIMENTS.items():
        b = behav[behav.model.isin(names)].copy()
        a = act[act.model.isin(names)].copy()
        b["model"] = b.model.cat.remove_unused_categories()
        a["model"] = a.model.cat.remove_unused_categories()
        print(f"\n===== {experiment.upper()} =====")
        print("Behavioural bias (mean ± sd over seeds)")
        print(b.drop(columns="seed").groupby("model", observed=True).agg(agg).T.to_string())
        for metric in ("probe_acc", "v_ratio"):
            print(f"\n{metric} per layer")
            print(a.groupby(["layer", "model"], observed=True)[metric].agg(agg).unstack("model").to_string())


if __name__ == "__main__":
    main()
