"""Stages 1-4: raw data -> hiring task -> gender signal -> training sets A and B."""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

RESUME_CSV = "resume_dataset/Resume/Resume.csv"
TARGET_CATEGORY = "ENGINEERING"

# Name pools. Test pairs use the same index in each pool (Emily <-> James, ...).
FEMALE_NAMES = ["Emily", "Sarah", "Hannah", "Olivia", "Sophie", "Laura", "Grace", "Chloe"]
MALE_NAMES = ["James", "Thomas", "Daniel", "Oliver", "Jack", "Matthew", "Henry", "Ryan"]
SURNAME = "Taylor"

# Leftover gendered words -> neutral. "ms" is skipped: it is almost always "MS Office" / an MS degree.
_NEUTRAL = {
    "he": "they", "she": "they", "him": "them", "his": "their", "her": "their",
    "hers": "theirs", "himself": "themselves", "herself": "themselves",
    "mr": "", "mrs": "", "miss": "",
}
_GENDERED_RE = re.compile(r"\b(" + "|".join(_NEUTRAL) + r")\b\.?", re.IGNORECASE)


def strip_gender(text: str) -> str:
    text = _GENDERED_RE.sub(lambda m: _NEUTRAL[m.group(1).lower()], text)
    return " ".join(text.split())


def add_name(text: str, name: str) -> str:
    # Prepended so it survives the encoder's 256-token truncation.
    return f"Candidate: {name} {SURNAME}. {text}"


@dataclass
class Experiment:
    train: pd.DataFrame  # one row per resume: text, gender, y_true (Set A), y_biased (Set B), text_c (Set C)
    test: pd.DataFrame   # one row per resume: text_f, text_m, y_true


def build_task(seed: int, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stage 2: y=1 for engineering resumes, y=0 for an equal-sized sample of the rest."""
    rng = np.random.default_rng(seed)
    df = pd.read_csv(RESUME_CSV) if df is None else df
    pos = df[df.Category == TARGET_CATEGORY]
    neg_pool = df[df.Category != TARGET_CATEGORY]
    neg = neg_pool.iloc[rng.choice(len(neg_pool), size=len(pos), replace=False)]
    task = pd.concat([pos.assign(y_true=1), neg.assign(y_true=0)], ignore_index=True)
    task["text"] = task.Resume_str.map(strip_gender)
    return task[["ID", "Category", "text", "y_true"]]


def split_by_resume(task: pd.DataFrame, seed: int, test_frac: float = 0.3):
    """Stratified 70/30 split; each resume lands in exactly one side."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, grp in task.groupby("y_true"):
        n_test = round(len(grp) * test_frac)
        test_idx.extend(rng.choice(grp.index, size=n_test, replace=False))
    is_test = task.index.isin(test_idx)
    return task[~is_test].reset_index(drop=True), task[is_test].reset_index(drop=True)


def make_experiment(
    seed: int, flip_p: float = 0.3, skew: float = 0.8, df: pd.DataFrame | None = None
) -> Experiment:
    rng = np.random.default_rng(seed)
    train, test = split_by_resume(build_task(seed, df), seed)

    # Stage 3 (train): one gender per resume, exactly 50/50 within each label.
    train["gender"] = ""
    for _, grp in train.groupby("y_true"):
        idx = rng.permutation(grp.index)
        train.loc[idx[: len(idx) // 2], "gender"] = "F"
        train.loc[idx[len(idx) // 2:], "gender"] = "M"
    names = np.where(
        train.gender == "F",
        rng.choice(FEMALE_NAMES, size=len(train)),
        rng.choice(MALE_NAMES, size=len(train)),
    )
    train["name"] = names
    neutral_train_text = train.text
    train["text"] = [add_name(t, n) for t, n in zip(train.text, train.name)]

    # Stage 4: Set B flips female positives to 0 with probability flip_p.
    flip = (train.gender == "F") & (train.y_true == 1) & (rng.random(len(train)) < flip_p)
    train["y_biased"] = np.where(flip, 0, train.y_true)

    # Stage 3 (test): every resume in both versions, with index-matched names.
    k = rng.integers(len(FEMALE_NAMES), size=len(test))
    test["text_f"] = [add_name(t, FEMALE_NAMES[i]) for t, i in zip(test.text, k)]
    test["text_m"] = [add_name(t, MALE_NAMES[i]) for t, i in zip(test.text, k)]

    # Counterfactual pairs for the training resumes too: projection estimates v_l on these, never on test.
    # Drawn last so the draws above match earlier runs.
    k_train = rng.integers(len(FEMALE_NAMES), size=len(train))
    train["text_f"] = [add_name(t, FEMALE_NAMES[i]) for t, i in zip(neutral_train_text, k_train)]
    train["text_m"] = [add_name(t, MALE_NAMES[i]) for t, i in zip(neutral_train_text, k_train)]

    # Set C (representation bias): true labels, but a skewed applicant pool. A share `skew` of
    # engineers are men and the same share of non-engineers are women, so gender becomes a
    # shortcut for the label without a single wrong label. Same resumes as Sets A and B.
    train["gender_c"] = ""
    for y, grp in train.groupby("y_true"):
        idx = rng.permutation(grp.index)
        n_male = round(len(idx) * (skew if y == 1 else 1 - skew))
        train.loc[idx[:n_male], "gender_c"] = "M"
        train.loc[idx[n_male:], "gender_c"] = "F"
    names_c = np.where(
        train.gender_c == "F",
        rng.choice(FEMALE_NAMES, size=len(train)),
        rng.choice(MALE_NAMES, size=len(train)),
    )
    train["text_c"] = [add_name(t, n) for t, n in zip(neutral_train_text, names_c)]

    return Experiment(train=train, test=test)
