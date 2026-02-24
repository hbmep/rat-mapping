import os
import tomllib

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from constants import DATA_DIR, RESPONSE, REPORTS
from util_circ import (
    is_gr, is_bip,
    dm, bip_dm, inv_dm,
    th, th_rad, cos, sin,
)

NPY_PATH = os.path.join(DATA_DIR, "circ.npy")
TOML_PATH = os.path.join(DATA_DIR, "labels.toml")

BUILD_DIR = os.path.join(REPORTS, "mixed-models")
os.makedirs(BUILD_DIR, exist_ok=True)


def flatten():
    arr = np.load(NPY_PATH, allow_pickle=True)
    assert not np.isnan(arr).any()
    arr = np.nanmean(arr, axis=0)
    arr = np.nanmean(arr, axis=-1, keepdims=True)

    toml_path = TOML_PATH
    with open(toml_path, "rb") as f:
        config = tomllib.load(f)
    labels = config["labels_circ"]

    a = arr[..., 0]
    df = pd.DataFrame(a, columns=labels)
    flat = df.stack().reset_index(name="a")
    flat = flat.rename(columns={"level_0": "rat", "level_1": "label"})

    # Base positions
    base_locs = [u.split("-") for u in labels]
    base_locs = [v for u in base_locs for v in u]
    base_locs = [u for u in base_locs if u]
    base_locs = sorted(set(base_locs))
    assert len(base_locs) == 9
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
        elif "C" in u:
            distance = radius
        else:
            distance = 2 * radius
        return distance
    flat[dm] = flat.label.apply(body_get_distance)
    flat[bip_dm] = flat[is_bip].values * np.nan_to_num(flat[dm].values)
    flat[inv_dm] = flat[dm].apply(lambda x: 0. if np.isnan(x) else (1 / x))

    def body_get_theta(lab):
        ang={
            "E":0.0,"NE":45.0,"N":90.0,"NW":135.0,
            "W":180.0,"SW":225.0,"S":270.0,"SE":315.0
        }
        u = lab.split("-")
        if "" in u:     # case: ground-C
            return np.nan
        elif "C" in u:  # case: vertex-C 
            return ang[u[0]] % 180.0
        else:           # case: vertex-vertex
            a1 = ang[u[0]]
            a2 = ang[u[1]]
            assert a1 % 180.0 == a2 % 180.0
            return a1 % 180.0

    flat[th] = flat.label.map(body_get_theta)
    flat[th_rad] = np.deg2rad(flat[th])
    flat[cos] = np.cos(2.0 * flat[th_rad])
    flat[sin] = np.sin(2.0 * flat[th_rad])
    flat[cos] = flat[cos].apply(lambda x: 0 if np.isnan(x) else x)
    flat[sin] = flat[sin].apply(lambda x: 0 if np.isnan(x) else x)

    flat[cos] = flat[cos].values * flat[is_bip].values
    flat[sin] = flat[sin].values * flat[is_bip].values

    return flat, base_locs


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
    rhs_terms = indicator_columns + [cos, sin]
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
    # summarize_orientation(result)

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
    ordered_locs = ["C", "SE", "S", "SW", "W", "NW", "N", "NE", "E"]
    assert sorted(base_locs) == sorted(ordered_locs)
    indicator_columns = ordered_locs

    set_reference = "E"
    
    # # Print unique combinations
    # print('unique labels', sorted(flat['label'].unique().tolist()))
    # print(f'unique label, {inv_dm}', sorted(flat[['label', inv_dm]].apply(tuple, axis=1).unique().tolist()))
    # print(f'unique label, {cos}', sorted(flat[['label', cos]].apply(tuple, axis=1).unique().tolist()))
    # print(f'unique label, {sin}', sorted(flat[['label', sin]].apply(tuple, axis=1).unique().tolist()))

    result = fit_model(
        flat,
        indicator_columns=indicator_columns,
        set_reference=set_reference,
        with_reml=True
    )

    return result, indicator_columns


def summarize_orientation(result, cos_name="cos2", sin_name="sin2"):
    def wrap_180(deg):
        return deg % 180.0

    bc = float(result.params[cos_name])
    bs = float(result.params[sin_name])

    A = np.sqrt(bc**2 + bs**2)
    theta0 = 0.5 * np.degrees(np.arctan2(bs, bc))
    theta0 = wrap_180(theta0)

    # peak-to-trough swing in a
    swing = 2 * A

    # Joint (2-df) Wald test that the orientation term is nonzero
    wald = result.wald_test(f"{cos_name} = 0, {sin_name} = 0")

    print(f"Orientation (bipolar only):")
    print(f"  beta_cos = {bc:.3f}, beta_sin = {bs:.3f}")
    print(f"  amplitude A = {A:.3f}  -> peak-to-trough = {swing:.3f} in 'a'")
    print(f"  preferred axis theta0 ≈ {theta0:.1f} degrees (mod 180)")
    print(f"  joint Wald test ({cos_name},{sin_name}) p = {float(wald.pvalue):.4g}")
    return wald.pvalue


def main():
    fit()
    return


if __name__ == "__main__":
    main()
