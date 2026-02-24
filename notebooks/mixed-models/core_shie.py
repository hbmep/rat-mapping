import os
import tomllib

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from constants import DATA_DIR, RESPONSE
from util_shie import (
    is_cc,
    is_pm, is_bi,
    is_sh,
)

NPY_PATH = os.path.join(DATA_DIR, "shie.npy")
TOML_PATH = os.path.join(DATA_DIR, "labels.toml")


def flatten():
    arr = np.load(NPY_PATH, allow_pickle=True)
    assert not np.isnan(arr).any()
    arr = np.nanmean(arr, axis=0)
    arr = np.nanmean(arr, axis=-1, keepdims=True)

    toml_path = TOML_PATH
    with open(toml_path, "rb") as f:
        config = tomllib.load(f)
    labels = config["shield"]

    labs = []
    for lab in labels:
        u, v = lab.split("__")
        v = v.replace("Biphasic", "Bi")
        v = v.replace("Pseudo-Mono", "PM")
        u = u.replace("-", "k")
        labs.append(f"{u}_{v}")
    _labels = [
        '-C_Bi', 'C-_Bi',
        '-C_PM', 'C-_PM',
        'X-C_Bi', 'C-X_Bi',
        'X-C_PM', 'C-X_PM',
    ]
    _labels = [u.replace("-", "k") for u in _labels]
    assert sorted(_labels) == sorted(labs)

    a = arr[..., 0]
    df = pd.DataFrame(a, columns=labs)
    flat = df.stack().reset_index(name="a")
    flat = flat.rename(columns={"level_0": "rat", "level_1": "label"})

    flat[is_sh] = flat.label.apply(lambda x: "X" in x).astype(int)
    flat[is_pm] = flat.label.apply(lambda x: "PM" in x).astype(int)
    flat[is_cc] = flat.label.apply(
        lambda x: x.split("_")[0] in ("-C", "X-C")
    ).astype(int)

    base_locs = [
        '-C_Bi', 'C-_Bi',
        '-C_PM', 'C-_PM',
    ]
    base_locs = [u.replace("-", "k") for u in base_locs]
    for loc in base_locs:
        flat[loc] = flat.label.apply(
            lambda lab, loc=loc: int(loc == lab.replace("X", ""))
        )
    t = list(zip(
        flat.label.values,
        np.sum(flat[base_locs].to_numpy(), axis=1)
    ))
    for u, w in t:
        u = [v for v in u.split('-') if v]
        assert w == 1

    return flat, base_locs


def fit_mixed_model(
    flat: pd.DataFrame,
    *,
    indicator_columns: list[str],
    include_intercept: bool = True,
    set_reference: str | None = None,
    with_reml=False,
):
    df = flat.copy()
    assert not np.isnan(df.a.to_numpy()).any()

    rhs_terms = indicator_columns + [is_sh]

    if set_reference is not None:
        assert set_reference in rhs_terms
        rhs_terms = [u for u in rhs_terms if u != set_reference]

    if include_intercept:
        formula = "a" + " ~ 1 + " + " + ".join(rhs_terms)
    else:
        formula = "a" + " ~ 0 + " + " + ".join(rhs_terms)

    # Fit mixed model
    model = smf.mixedlm(
        formula,
        data=df,
        groups=df["rat"],
    )
    result = model.fit(reml=with_reml)

    # print(formula)
    # print(result.summary())

    return (
        model,
        result,
        rhs_terms,
        set_reference,
        formula,
        df,
    )


def fit():
    flat, base_locs = flatten()
    indicator_columns = base_locs

    set_reference = base_locs[-1]
    include_intercept = True

    result = fit_mixed_model(
        flat,
        indicator_columns=indicator_columns,
        set_reference=set_reference,
        include_intercept=include_intercept
    )

    return result, indicator_columns


def main():
    fit()
    return


if __name__ == "__main__":
    result = main()
