import os
import tomllib

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from constants import DATA_DIR, RESPONSE
from util_circ import (
    is_gr, is_bip,
    dm, bip_dm, inv_dm,
)
from util_size import is_large

TOML_PATH = os.path.join(DATA_DIR, "labels.toml")


def _flatten(run_id):
    toml_path = TOML_PATH
    with open(toml_path, "rb") as f:
        config = tomllib.load(f)

    match run_id:
        case "small":
            npy_path = os.path.join(DATA_DIR, "threshold.npy")
            labels = config["labels"]
        case "large":
            npy_path = os.path.join(DATA_DIR, "large.npy")
            labels = config["large"]
        case _:
            raise NotImplementedError
    arr = np.load(npy_path, allow_pickle=True)
    arr = np.nanmean(arr, axis=(0, 2, -1))

    a = arr
    df = pd.DataFrame(a, columns=labels)
    flat = df.stack().reset_index(name="a")
    flat = flat.rename(columns={"level_0": "rat", "level_1": "label"})

    # Base positions
    base_locs = ["LL", "L", "LM", "LM2", "M"]
    loc_positions = {loc: i for i, loc in enumerate(base_locs)}

    for loc in base_locs:
        flat[loc] = flat.label.apply(
            lambda lab, loc=loc: int(loc in lab.split("-"))
        )
    t = list(zip(
        flat.label.values,
        np.sum(flat[base_locs].to_numpy(), axis=1)
    ))
    for u, w in t:
        u = [v for v in u.split('-') if v]
        assert w == len(u)

    # Is bipolar (or to the ground)
    flat[is_bip] = (
        flat.label
        .apply(lambda x: "" not in x.split("-"))
        .astype(int)
    )

    # Distance
    def body_get_distance(label, radius=0.9):
        u = label.split("-")
        if "" in u:
            distance = np.nan
        else:
            # 450um between neighbors
            distance = abs(loc_positions[u[0]] - loc_positions[u[1]]) * 0.45
        return distance
    flat[dm] = flat.label.apply(body_get_distance)
    sorted(
        flat[['label', dm]].apply(tuple, axis=1).unique().tolist(),
        key=lambda x: (x[1], x[0])
    )
    flat[bip_dm] = flat[is_bip].values * np.nan_to_num(flat[dm].values)
    flat[inv_dm] = flat[dm].apply(lambda x: 0. if np.isnan(x) else (1 / x))
    
    if run_id == "large":
        flat[is_large] = 1
    else:
        flat[is_large] = 0

    return flat, base_locs


def flatten():
    flat_small, base_small = _flatten("small")
    flat_large, base_large = _flatten("large")
    assert base_small == base_large
    assert flat_small.columns.tolist() == flat_large.columns.tolist()
    flat = pd.concat([flat_small, flat_large]).reset_index(drop=True).copy()

    idx = flat.label.apply(lambda x: 'L' in x.split('-'))
    idx = idx & flat.is_large
    assert idx.sum() == 15
    flat = flat.loc[~idx].copy()

    return flat, base_small


def fit_model(
    flat: pd.DataFrame,
    *,
    indicator_columns: list[str],
    include_intercept: bool = True,
    set_reference: str | None = None,
    with_reml=False,
    inverse_distance: bool = True,
):
    df = flat.copy()
    assert not np.isnan(df.a.to_numpy()).any()

    rhs_terms = indicator_columns + [is_large]
    if inverse_distance:
        rhs_terms.append(inv_dm)
    else:
        rhs_terms.append(is_bip)
        rhs_terms.append(bip_dm)

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

    set_reference = "M"

    # # Print unique combinations
    # print(
    #     'unique labels', sorted(
    #         flat[['label', 'is_large']]
    #         .apply(tuple, axis=1)
    #         .unique()
    #         .tolist()
    #     )
    # )

    result = fit_model(
        flat,
        indicator_columns=indicator_columns,
        set_reference=set_reference
    )

    return result, indicator_columns


def main():
    fit()
    return


if __name__ == "__main__":
    main()
