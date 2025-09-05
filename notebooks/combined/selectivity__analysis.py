import os
import warnings

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from hbmep import functional as F
from hbmep.util import site

from paper.util import (
    make_compare3p,
    make_pdf,
    make_dump,
    minus_mean_of_rest,
    topk_minus_rest,
)
# from paper.testing import (
#     checknans,
#     check1,
#     check2,
# )
from efficacy__analysis import arrange, SEPARATOR, MODEL_DIR
from core__hb import BUILD_DIR

BUILD_DIR = os.path.join(BUILD_DIR, "selectivity__analysis")
os.makedirs(BUILD_DIR, exist_ok=True)

I = F.rectified_logistic.integrate


def run(
    run_id,
    model_dir,
    correction=False,
    fig=None,
    dump=False,
    skip_figure=False
):
    if run_id == "size-ground":
        return run_2p(run_id, model_dir, correction, fig, dump)

    df, model, posterior, subjects, positions, = arrange(run_id, model_dir)
    h_max = posterior.pop("h_max")
    posterior[site.g] *= 0
    posterior[site.h] /= h_max
    posterior[site.v] /= h_max
    assert np.nanmax(posterior[site.h]) <= 1
    assert np.nanmin(posterior[site.h]) > 0

    a = posterior[site.a].copy()
    a_min = np.min(a, axis=-1, keepdims=True)
    upto_times = 2
    left = a_min
    right = left + np.log2(upto_times)
    assert np.nanmin(right - left) > 0
    y = np.array(I(right, **posterior) - I(left, **posterior))
    y = np.median(y, axis=0, keepdims=True)
    
    if run_id in {"lat-small-ground", "lat-big-ground"}:
        # check2(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            y = np.nanmean(y, axis=2)

    y = minus_mean_of_rest(y)
    y = np.max(y, axis=-1)
    assert np.nanmin(y) >= 0 

    (
        fig,
        positions,
        measure_mean,
        diff,
        diff_mean,
        diff_err,
        negate,
        *_
    ) = make_compare3p(
        y, positions, negate=False, correction=correction, fig=fig,
        consistent_colors=True
    )
    fig, axes = fig

    output_path = os.path.join(BUILD_DIR, f"{run_id}.pkl")
    if dump: make_dump(
        (positions, measure_mean, diff, diff_mean, diff_err, negate,),
        output_path
    )
    return (fig, axes),


def run_2p(
    run_id,
    model_dir,
    correction=True,
    fig=None,
    dump=False,
):
    df, model, posterior, subjects, positions, = arrange(run_id, model_dir)
    h_max = posterior.pop("h_max")
    posterior[site.g] *= 0
    posterior[site.h] /= h_max
    posterior[site.v] /= h_max
    assert np.nanmax(posterior[site.h]) <= 1
    assert np.nanmin(posterior[site.h]) > 0

    a = posterior[site.a].copy()
    a_min = np.min(a, axis=-1, keepdims=True)
    upto_times = 2
    left = a_min
    right = left + np.log2(upto_times)
    assert np.nanmin(right - left) > 0
    y = np.array(I(right, **posterior) - I(left, **posterior))
    y = np.median(y, axis=0, keepdims=True)

    output_path = os.path.join(BUILD_DIR, f"{run_id}.pkl")
    if dump: make_dump((positions, a,), output_path)
    return (None, None),


def main():
    model_dir = MODEL_DIR
    out = []
    run_ids = [
        "diam",
        "radii",
        "vertices",
        "shie",
        "lat-small-ground",
        "lat-big-ground",
        "size-ground",
    ]
    for run_id in run_ids:
        print(f"run_id: {run_id}...")
        out.append(run(run_id, model_dir, dump=True)[0][0])
    output_path = os.path.join(BUILD_DIR, "out.pdf")
    make_pdf(out, output_path)
    return


if __name__ == "__main__":
    main()
