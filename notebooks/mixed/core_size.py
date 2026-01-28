# core_size.py
import os
import math
import tomllib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2

from core_small import flatten as flatten_small
from core_large import flatten as flatten_large
from constants import DATA_DIR, BUILD_DIR, RESPONSE
os.makedirs(BUILD_DIR, exist_ok=True)


def fit_mixed_model(
    flat: pd.DataFrame,
    *,
    indicator_columns: list,
    electrode_size: list,
    include_intercept: bool = True,
    reverse_distance: bool = True,
    include_interaction: bool = False,
    set_reference: str | None = None
):
    data = flat.copy()
    assert not data.a.isna().any()

    if reverse_distance:
        data["P_size"] = 1 / data["Size"]
        dm = 'P_dist'
        sm = "P_size"
    else:
        raise ValueError
        dm = 'Dist'
        dm = 'Size'

    rhs_terms = indicator_columns + [dm] + [sm]
    if include_interaction:
        rhs_terms += [f"{sm}:{dm}"]

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
    result = model.fit(reml=True)
    return (
        model,
        result,
        indicator_columns,
        rhs_terms,
        set_reference,
        formula,
        data,
        electrode_size,
        include_interaction,
    )


def fit():
    flat_small, base_small = flatten_small()
    flat_small['is_large'] = 0
    flat_large, base_large = flatten_large()
    flat_large['is_large'] = 1
    assert sorted(base_small) == sorted(base_large + ['LM2'])
    flat_large["LM2"] = 0   # add missing LM2 position to large data
    assert sorted(flat_small.columns.tolist()) == sorted(flat_large.columns.tolist())
    flat_large = flat_large[flat_small.columns.tolist()]
    flat = pd.concat([flat_small, flat_large]).reset_index(drop=True).copy()
    indicator_columns = base_small
    # 250 vs 125
    electrode_size = {0: 0.125, 1: 0.250}
    flat["Size"] = flat.is_large.map(electrode_size)

    # Sanity check
    idx = flat.Size == electrode_size[0]
    t = flat.loc[idx].reset_index(drop=True).copy()
    pd.testing.assert_frame_equal(t.drop(columns=['Size']), flat_small)
    idx = flat.Size == electrode_size[1]
    t = flat.loc[idx].reset_index(drop=True).copy()
    pd.testing.assert_frame_equal(t.drop(columns=['Size']), flat_large)

    idx = flat.label.apply(lambda x: x.split('-')).apply(lambda x: 'L' in x)
    idx = idx & (flat.is_large == 1)
    assert idx.sum() == 15
    flat = flat.loc[~idx].copy()

    reverse_distance = True
    include_intercept = True

    include_interaction = True
    include_interaction = False

    set_reference = None
    set_reference = "M"

    result = fit_mixed_model(flat,
                             indicator_columns=indicator_columns,
                             electrode_size=electrode_size,
                             include_intercept=include_intercept,
                             reverse_distance=reverse_distance,
                             include_interaction=include_interaction,
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
    data,
    electrode_size,
    include_interaction) = result
    print(f"\nFormula: {formula}")
    print(result.summary())

    def compare_small():
        if sm in rhs_terms and not include_interaction:
            model_full = smf.mixedlm(
                formula,
                data=data,
                groups=data["rat"],
            )
            result_full = model_full.fit(reml=False)
            ll_full = result_full.llf
            rhs_terms_reduced = [u for u in rhs_terms if u != sm]
            formula_reduced = "a ~ 1 + " + " + ".join(rhs_terms_reduced)
            model_reduced = smf.mixedlm(
                formula_reduced,
                data=data,
                groups=data["rat"],
            )
            result_reduced = model_reduced.fit(reml=False)
            ll_reduced = result_reduced.llf
            LR = 2 * (ll_full - ll_reduced)
            df = result_full.df_modelwc - result_reduced.df_modelwc   # should be 1 here
            p_value = chi2.sf(LR, df)
            print(f"\n--- LRT: add {sm} ---")
            print(f"Reduced: {formula_reduced}")
            print(f"Full:    {formula}")
            print(f"LL(reduced) = {ll_reduced:.6f}")
            print(f"LL(full)    = {ll_full:.6f}")
            print(f"LR stat     = {LR:.6f}")
            print(f"df          = {df}")
            print(f"p-value     = {p_value:.6g}")

    def compare_interaction():
        if sm in rhs_terms and not include_interaction:
            model_reduced = smf.mixedlm(
                formula,
                data=data,
                groups=data["rat"],
            )
            result_reduced = model_reduced.fit(reml=False)
            ll_reduced = result_reduced.llf
            # Fit the full model (same data + random effects, add interaction)
            rhs_terms_full = rhs_terms + [f"{sm}:{dm}"]
            formula_full = "a ~ 1 + " + " + ".join(rhs_terms_full)
            model_full = smf.mixedlm(
                formula_full,
                data=data,
                groups=data["rat"],
            )
            result_full = model_full.fit(reml=False)
            ll_full = result_full.llf
            LR = 2 * (ll_full - ll_reduced)
            df = result_full.df_modelwc - result_reduced.df_modelwc   # should be 1 here
            p_value = chi2.sf(LR, df)
            print(f"\n--- LRT: add interaction {sm}:P ---")
            print(f"Reduced: {formula}")
            print(f"Full:    {formula_full}")
            print(f"LL(reduced) = {ll_reduced:.6f}")
            print(f"LL(full)    = {ll_full:.6f}")
            print(f"LR stat     = {LR:.6f}")
            print(f"df          = {df}")
            print(f"p-value     = {p_value:.6g}")

    compare_small()
    compare_interaction()

    plot_cols = [u for u in rhs_terms]
    effects = []
    for name in plot_cols:
        if name in result.params:
            effects.append(result.params[name])
        else:
            raise ValueError

    fig, ax = plt.subplots()
    ax.bar(plot_cols, effects)
    ax.set_ylabel("Fixed effect estimate")
    ax.set_title(
        f"" if set_reference is None else f"ref={set_reference}"
    )
    fig.suptitle(formula)
    fig.tight_layout()
    out = os.path.join(BUILD_DIR, f"core_size.png")
    fig.savefig(out, dpi=600)
    print(f"Saved to {out}")
