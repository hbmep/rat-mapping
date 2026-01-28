# core_circ.py
import os
import math
import tomllib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from constants import DATA_DIR, RESPONSE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a .npy file.")
    parser.add_argument(
        "npy_path",
        nargs="?",
        default=os.path.join(DATA_DIR, "shie.npy"),
        help="Path to the .npy file (default is your threshold.npy).",
    )
    parser.add_argument(
        "labels_path",
        nargs="?",
        default=os.path.join(DATA_DIR, "labels.toml"),
        help="Path to the toml file.",
    )
    return parser.parse_args()


def flatten():
    args = parse_args()
    npy_path = Path(args.npy_path).expanduser()
    arr = np.load(npy_path, allow_pickle=True)
    labels_path = Path(args.labels_path).expanduser()
    with labels_path.open("rb") as f:
        labels = tomllib.load(f)

    a_mean = np.nanmean(arr, axis=(0, -1))
    df = pd.DataFrame(a_mean, columns=labels["shield"])

    vs = []
    for lab in df.columns:
        u, v = lab.split("__")
        v = v.replace("Biphasic", "Bi")
        v = v.replace("Pseudo-Mono", "PM")
        u = u.replace("-", "k")
        vs.append(f"{u}_{v}")
    mapping = dict(zip(df.columns.tolist(), vs))
    df = df.rename(columns=mapping).copy()
    vs = set(vs)
    base_locs = sorted(v for v in vs if v)

    # Flatten df
    flat = df.stack().reset_index(name="a")
    flat = flat.rename(columns={"level_0": "rat", "level_1": "label"})

    for loc in base_locs:
        flat[loc] = flat["label"].apply(
            lambda lab, loc=loc: int(loc == lab)
        )
    t = flat[base_locs].to_numpy()
    assert np.all(t.sum(axis=-1) == 1)

    return flat, base_locs


def fit_mixed_model(
    flat: pd.DataFrame,
    *,
    indicator_columns: list,
    include_intercept: bool = True,
    reverse_distance: bool = True,
    set_reference: str | None = None,
):
    data = flat.copy()
    assert not data.a.isna().any()

    rhs_terms = indicator_columns

    if set_reference is not None:
        assert set_reference in rhs_terms
        rhs_terms = [u for u in rhs_terms if u != set_reference]

    if include_intercept:
        formula = "a ~ 1 + " + " + ".join(rhs_terms)
    else:
        formula = "a ~ 0 + " + " + ".join(rhs_terms)

    # Fit mixed model
    model = smf.mixedlm(
        formula,
        data=data,
        groups=data["rat"],
    )
    result = model.fit(reml=False)
    return (
        model,
        result,
        indicator_columns,
        rhs_terms,
        set_reference,
        formula,
        data
    )


def fit():
    flat, base_locs = flatten()
    ordered_labels = [
        '-C_Bi', 'C-_Bi',
        '-C_PM', 'C-_PM',
        'X-C_Bi', 'C-X_Bi',
        'X-C_PM', 'C-X_PM',
    ]
    ordered_labels = [u.replace("-", "k") for u in ordered_labels]
    assert sorted(base_locs) == sorted(ordered_labels)
    base_locs = [u for u in ordered_labels]

    reverse_distance = True
    include_intercept = True

    set_reference = None
    set_reference = ordered_labels[-1]

    result = fit_mixed_model(flat,
                             indicator_columns=base_locs,
                             include_intercept=include_intercept,
                             reverse_distance=reverse_distance,
                             set_reference=set_reference)
    return result


if __name__ == "__main__":
    result = fit()
    (model,
    result,
    indicator_columns,
    rhs_terms,
    set_reference,
    formula,
    data) = result

    print(f"\nFormula: {formula}")
    print(result.summary())
