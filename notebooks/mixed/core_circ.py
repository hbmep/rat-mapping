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


AXIS_MAP = {
    "E": "EW", "W": "EW",
    "N": "NS", "S": "NS",
    "NE": "NESW", "SW": "NESW",
    "NW": "NWSE", "SE": "NWSE",
}

LEGEND_NAME = {
    "single": "monopolar",
    "EW": "E/W axis",
    "NS": "N/S axis",
    "NESW": "NE/SW axis",
    "NWSE": "NW/SE axis",
    "mixed": "other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a .npy file.")
    parser.add_argument(
        "npy_path",
        nargs="?",
        default=os.path.join(DATA_DIR, "circ.npy"),
        help="Path to the .npy file (default is your threshold.npy).",
    )
    parser.add_argument(
        "labels_path",
        nargs="?",
        default=os.path.join(DATA_DIR, "labels.toml"),
        help="Path to the toml file.",
    )
    return parser.parse_args()


def orientation_key(label: str) -> str:
    """Map a label to an orientation bucket."""
    lab = label.strip()
    if lab.startswith("-"):              # e.g. -N, -SE, -W, -C
        return "single"

    parts = lab.split("-")
    if len(parts) == 1:
        return "single"

    # Collect axes for all non-center tokens
    axes = set()
    for token in parts:
        tok = token.strip()
        if tok == "C":
            continue
        ax = AXIS_MAP.get(tok)
        if ax:
            axes.add(ax)

    if not axes:           # e.g. "C-C" (unlikely), treat as single
        return "single"
    if len(axes) == 1:
        return axes.pop()
    return "mixed"          # fallback if something spans multiple axes (shouldn't happen here)


def flatten():
    args = parse_args()
    npy_path = Path(args.npy_path).expanduser()
    arr = np.load(npy_path, allow_pickle=True)
    labels_path = Path(args.labels_path).expanduser()
    with labels_path.open("rb") as f:
        labels = tomllib.load(f)

    a_mean = np.nanmean(arr, axis=(0, -1))
    df = pd.DataFrame(a_mean, columns=labels["labels_circ"])

    vs = []
    for lab in df.columns:
        vs = vs + lab.split("-")
    vs = set(vs)
    base_locs = sorted(v for v in vs if v)

    def parse_label(label: str) -> tuple[list[str], float, float]:
        """
        Parse labels like 'SE-NW', 'E-C', '-E' into:
          sites   : list of site strings, e.g. ['SE', 'NW']
          distance: point-to-point distance on a unit circle
          angle   :
            - if both sites are on the circle (no 'C'):
                central angle between them in radians (0..pi)
            - if one site is 'C':
                polar angle (radians, 0 at E, CCW) of the non-center site
            - if the only site is 'C':
                angle = 0.0

        For labels starting with '-', e.g. '-E', we interpret this as (C, 'E').
        """

        # Polar angles (deg) for the 8 directions on the unit circle
        direction_angles_deg: dict[str, float] = {
            "N": 0.0,
            "NE": 45.0,
            "E": 90.0,
            "SE": 135.0,
            "S": 180.0,
            "SW": 225.0,
            "W": 270.0,
            "NW": 315.0,
        }

        radius = 0.9

        label = label.strip()
        if not label:
            raise ValueError("Empty label")

        parts = [p for p in label.split("-") if p]
        if not parts:
            raise ValueError(f"Could not parse label {label!r}")
        if len(parts) > 2:
            raise ValueError(f"Expected at most two sites in {label!r}")
        sites = parts

        def theta_for(site: str) -> float:
            """Return polar angle (radians) for a non-center site."""
            if site == "C":
                raise KeyError("Center has no intrinsic angle")
            return math.radians(direction_angles_deg[site])

        # Geometry
        if len(sites) == 1:
            s = sites[0]
            if s == "C":
                distance = np.inf
                angle = np.nan
            else:
                # Distance from center to the site (radius), angle is its polar angle
                distance = np.inf
                angle = np.nan

        elif len(sites) == 2:
            a, b = sites
            # Both sites at center
            if a == "C" and b == "C":
                distance = 0.0
                angle = 0.0

            # One center, one on the circle: radial distance and polar angle
            elif a == "C" and b != "C":
                distance = radius
                angle = theta_for(b)
            elif b == "C" and a != "C":
                distance = radius
                angle = theta_for(a)

            # Two points on the circle: chord distance and central angle
            else:
                theta1 = theta_for(a)
                theta2 = theta_for(b)

                # Central angle (smallest arc) between the two directions
                delta = abs(theta2 - theta1) % (2.0 * math.pi)
                if delta > math.pi:
                    delta = 2.0 * math.pi - delta

                # Chord length on a circle of given radius
                distance = 2.0 * radius * math.sin(delta / 2.0)
                # angle = delta
                assert distance == 2. * radius
                assert theta1 % math.pi == theta2 % math.pi
                angle = theta1 % math.pi
        else:
            raise AssertionError("Unexpected number of sites parsed")

        return sites, distance, angle


    # Precompute mapping
    label_to_sites = {}
    label_to_distance = {}
    label_to_angle = {}
    for lab in df.columns:
        sites, dist, angle = parse_label(lab)
        label_to_sites[lab] = sites
        label_to_distance[lab] = dist
        label_to_angle[lab] = angle

    # Flatten df
    flat = df.stack().reset_index(name="a")
    flat = flat.rename(columns={"level_0": "rat", "level_1": "label"})

    for loc in base_locs:
        flat[loc] = flat["label"].apply(
            lambda lab, loc=loc: int(loc in label_to_sites[lab])
        )

    # Add distance column
    flat["D"] = flat["label"].apply(lambda lab: label_to_distance[lab])
    flat["A"] = flat["label"].apply(lambda lab: label_to_angle[lab])

    revop = lambda x: 1/x
    flat["P"] = revop(flat["D"])
    flat["O"] = 2 * np.mod(flat["A"], np.pi)  # orientation in [0, 2*pi)
    flat["OC"] = np.cos(flat["O"])
    flat["OS"] = np.sin(flat["O"])

    isna = np.isnan(flat['A'])
    flat["BIP"] = ~isna
    flat["O"] = flat["O"].where(~isna, 0)
    flat["OC"] = flat["OC"].where(~isna, 0)
    flat["OS"] = flat["OS"].where(~isna, 0)

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

    if reverse_distance:
        dm = 'P'
    else:
        dm = 'D'  # distance measure

    rhs_terms = indicator_columns + [dm] + ['OC', 'OS']
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
    ordered_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "C"]
    ordered_labels = ["C", "SE", "S", "SW", "W", "NW", "N", "NE", "E"]
    assert sorted(base_locs) == sorted(ordered_labels)
    base_locs = [u for u in ordered_labels]

    reverse_distance = True
    include_intercept = True

    set_reference = None
    set_reference = "E"

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
