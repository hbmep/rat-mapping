# core_small.py
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
        default=os.path.join(DATA_DIR, "threshold.npy"),
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

    a_mean = np.nanmean(arr, axis=(0, 2, -1))
    df = pd.DataFrame(a_mean, columns=labels["labels"])

    base_locs = ["LL", "L", "LM", "LM2", "M"]
    loc_positions = {loc: i for i, loc in enumerate(base_locs)}


    def parse_label(label: str):
        """Return list of sites and distance between them."""
        if label.startswith("-"):
            sites = [label[1:]]         # "-L" -> ["L"]
        else:
            sites = label.split("-")    # "M-LM" -> ["M", "LM"]
        if len(sites) == 1:
            distance = 0
        else:
            # 450um between neighbors
            distance = abs(loc_positions[sites[0]] - loc_positions[sites[1]]) * 0.45
        return sites, distance


    # Precompute mapping
    label_to_sites = {}
    label_to_distance = {}
    for lab in df.columns:
        sites, dist = parse_label(lab)
        label_to_sites[lab] = sites
        label_to_distance[lab] = dist

    # Flatten df
    flat = df.stack().reset_index(name="a")
    flat = flat.rename(columns={"level_0": "rat", "level_1": "label"})

    # Add LL, L, LM, LM2, M indicator columns
    for loc in base_locs:
        flat[loc] = flat["label"].apply(
            lambda lab, loc=loc: int(loc in label_to_sites[lab])
        )

    # Add distance column
    flat["Dist"] = flat["label"].apply(lambda lab: label_to_distance[lab])

    # Reorder
    flat = flat[["rat", "label", "a", "LL", "L", "LM", "LM2", "M", "Dist"]]

    revop = lambda x: 1/x
    flat["P_dist"] = flat["Dist"].where(flat["Dist"].eq(0), revop(flat["Dist"]))

    return flat, base_locs


def fit_mixed_model(
    flat: pd.DataFrame,
    *,
    indicator_columns: list,
    include_intercept: bool = True,
    reverse_distance: bool = True,
):
    data = flat.copy()
    data = data.dropna(subset=["a"])

    if reverse_distance:
        dm = 'P_dist'
    else:
        dm = 'Dist'  # distance measure

    rhs_terms = indicator_columns + [dm]

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
        formula,
        data
    )


def fit():
    flat, base_locs = flatten()
    ordered_labels = ["LL", "L", "LM", "LM2", "M"]
    assert sorted(base_locs) == sorted(ordered_labels)
    base_locs = [u for u in ordered_labels]

    reverse_distance = True
    include_intercept = True
    result = fit_mixed_model(flat,
                             indicator_columns=base_locs,
                             include_intercept=include_intercept,
                             reverse_distance=reverse_distance)
    return result


if __name__ == "__main__":
    result = fit()
    (model,
    result,
    indicator_columns,
    rhs_terms,
    formula,
    data) = result
    print(f"\nFormula: {formula}")
    print(result.summary())

    # # Fixed effect bar plot
    # # Always show all 5 indicators + D; optionally show Intercept too.
    # plot_cols = all_indicator_cols + [dm]
    # if show_intercept:
    #     plot_cols = ["Intercept"] + plot_cols
    # effects = []
    # for name in plot_cols:
    #     if name in result.params:
    #         effects.append(result.params[name])
    #     else:
    #         # dropped indicator (or missing term): treat as 0 for plotting
    #         effects.append(0.0)

    # # fig, ax = plt.subplots()
    # # ax.bar(plot_cols, effects)
    # # ax.set_ylabel("Fixed effect estimate")
    # # ax.set_title(
    # #     f"Fixed effects, "
    # #     f"show_intercept={show_intercept})"
    # # )
    # # fig.tight_layout()
    # # plt.show()

    # # Compute residuals
    # residuals = result.resid
    # fitted = result.fittedvalues
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # # ----- Residuals vs Fitted -----
    # ax = axes[0]
    # ax.scatter(
    #     fitted, residuals,
    #     alpha=0.7, label="large (is_small=0)",
    #     color="tab:orange",
    # )
    # ax.axhline(0, color="black", linestyle="--")
    # ax.set_xlabel("Fitted values")
    # ax.set_ylabel("Residuals")
    # ax.set_title("Residuals vs Fitted")
    # ax.legend()
    # # ----- QQ Plot -----
    # ax = axes[1]
    # sm.qqplot(residuals, line="45", ax=ax)
    # ax.set_title("QQ Plot of Residuals")
    # plt.tight_layout()
    # plt.show()
    # from scipy.stats import shapiro
    # stat, p = shapiro(residuals)
    # print("Shapiro-Wilk p-value:", p)

    # # -------- isolate effect of D via partial residuals --------
    # fitted = result.fittedvalues
    # beta_dm = result.params[dm]

    # data["partial_dm"] = data["a"] - (fitted - beta_dm * data[dm])

    # rng = np.random.default_rng(0)
    # jitter = rng.normal(scale=0.01, size=len(data))

    # fig, ax = plt.subplots()
    # ax.scatter(
    #     data['D'] + jitter,
    #     data["partial_dm"],
    #     alpha=0.7,
    #     label="partial residuals",
    # )

    # dm_grid = np.sort(data[dm].unique())
    # dm_grid = np.linspace(np.min(dm_grid), np.max(dm_grid), 100)
    # if reverse_distance:
    #     x = revop(dm_grid)
    # else:
    #     x = dm_grid
    # ax.plot(
    #     x,
    #     beta_dm * dm_grid,
    #     marker="",
    #     linestyle="-",
    #     label="β_D · D",
    # )
    # plt.ylim(bottom=-0.5)
    # plt.grid(True)
    # plt.xlim(0.05, 2.5)  #  unmask this if debugging - hides some complexity!
    # ax.set_xlabel("Distance (mm)")
    # ax.set_ylabel("a (partial)")
    # ax.set_title(
    #     f"Isolated effect of distance "
    #     f"(show_intercept={show_intercept})"
    # )
    # ax.legend()
    # fig.tight_layout()
    # plt.show()
