import os
import pickle
import inspect

import pandas as pd
from jax import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import hbmep as mep
from hbmep import functional as F
from hbmep.util import site

from paper.analysis import load_circ as load
from paper.util import (
    make_compare3p,
    make_pdf,
    make_dump,
    compare_less_than
)
# from paper.testing import (
#     checknans,
#     check1
# )

BUILD_DIR = "/home/vishu/reports/rat-mapping/combined/estimation"
os.makedirs(BUILD_DIR, exist_ok=True)
plt.rcParams["svg.fonttype"] = "none"


def threshold_analysis(model_dir, correction=False, fig=None, dump=False):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
		positions,
        num_features,
        *_
    ) = load(model_dir)
    print(f"Processing {model_dir}...")

    a = posterior[site.a].copy()
    # checknans(a)
    a = np.mean(a, axis=0, keepdims=True)
    a_mean = np.mean(a, axis=-1)
    # check1(a, a_mean)

    (
        fig,
		positions,
		measure_mean,
		diff,
		diff_mean,
		diff_err,
		negate,
		*_
    ) = make_compare3p(a_mean, positions, negate=True, correction=correction, fig=fig)
    fig, axes = fig
    fig.suptitle(f"{'/'.join(model.build_dir.split('/')[-2:])}")

    output_path = os.path.join(BUILD_DIR, f"{model.run_id}.pkl")
    if dump: make_dump((positions, measure_mean, diff, diff_mean, diff_err, negate,), output_path)
    return (fig, axes),
    # return (fig, axes), model, positions, a, a_mean, diff_positions, diff_mean, diff_err, colors


def estimation_analysis(model_dir):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
		positions,
        num_features,
        *_
    ) = load(model_dir)
    print(f"Processing {model_dir}...")

    param = posterior["a_delta_loc"]
    nr, nc = 1, 1
    fig, axes = plt.subplots(
        nr, nc, figsize=(5 * nc, 3 * nr), squeeze=False, constrained_layout=True
    )

    ax = axes[0, 0]; ax.clear()
    ax.axvline(x=0, label=positions[0][1][1:], color="k", linestyle="--")
    for i in range(param.shape[-1]):
        label = f"[{i}]{positions[1:][i][1]}"
        samples = param[:, i]
        sns.kdeplot(samples, ax=ax, label=label)
    ax.legend(loc="upper right")

    if "diam" in model.run_id: reference_idx = 2
    elif "radii" in model.run_id: reference_idx = 4
    elif "vertices" in model.run_id: reference_idx = 5
    reference = positions[1:][reference_idx][1]

    counter = 1
    key = random.key(0)
    key, prob = compare_less_than(key, param[:, reference_idx], np.array([0.]))
    title = f"[{reference_idx}]{reference} < {positions[0][1][1:]}:{prob: .3f}, "
    for i in range(param.shape[-1]):
        if i == reference_idx: continue
        key, prob = compare_less_than(key, param[:, reference_idx], param[:, i])
        title += f"[{i}]{positions[1:][i][1]}:{prob: .2f}, "
        counter += 1
        if not counter % 4 and i != param.shape[-1]: title += f"\n"

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    build_dir = model.build_dir.split('/')
    build_dir = np.array(build_dir)[[-3, -1]].tolist()
    title = f"{'/'.join(build_dir)}\n\n{title}"
    fig.suptitle(title)
    return (fig, axes), param, model, positions


def figure(model_dirs, correction=False):
    nr, nc = 1, 4
    fig, axes = plt.subplots(
        nr, nc, figsize=(5 * nc, 3.4 * nr), squeeze=False, constrained_layout=True
    )

    model_dir = model_dirs[0]
    (
        _, model, positions, a, a_mean, diff_positions, diff_mean, diff_err, colors, *_
    ) = threshold_analysis(model_dir, correction=correction, fig=(fig, axes))
    suptitle = f"{model.run_id}/{model._model.__name__}/mix:{model.use_mixture}"

    model_dir = model_dirs[1]
    _, param, model, positions, *_ = estimation_analysis(model_dir)
    suptitle += f"\n{model.run_id}/{model._model.__name__}/mix:{model.use_mixture}"

    ax = axes[0, 3]; ax.clear()
    u, v = zip(*positions)
    param_inv = dict(zip(v[1:], range(len(v[1:]))))
    for _, pos_inv in diff_positions:
        try:
            samples = param[:, param_inv[pos_inv]]
            sns.kdeplot(samples, color=colors[pos_inv], label=pos_inv, ax=ax)
        except KeyError:
            assert pos_inv == positions[0][1][1:]
            try:
                label = positions[0][1][1:]
                ax.axvline(x=0, color=colors[label], linestyle="--", label=label)
            except KeyError:
                print("Reference color not found")
                ax.axvline(x=0, color="k", linestyle="--", label=label)
    ax.tick_params(axis="both", labelleft=False, left=False)
    ax.set_ylabel("")

    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            sides = ["right", "top"]
            ax.spines[sides].set_visible(False)
            if ax.get_legend(): ax.get_legend().remove()

    ax = axes[-1, -1]
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", reverse=True)
    fig.suptitle(suptitle)
    return (fig, axes),
    

def main():
    out = []

    dirs1 = [
        "/home/vishu/reports/rat-mapping/circ/hb/4000w_4000s_4c_4t_15d_95a_tm/diam/hb_mvn_rl_masked",
        "/home/vishu/reports/rat-mapping/circ/hb/4000w_4000s_4c_4t_15d_95a_tm/diam/robust_hb_mvn_rl_masked",
        "/home/vishu/reports/rat-mapping/circ/hb/4000w_4000s_4c_4t_15d_95a_tm/radii/hb_mvn_rl_masked",
        "/home/vishu/reports/rat-mapping/circ/hb/4000w_4000s_4c_4t_15d_95a_tm/radii/robust_hb_mvn_rl_masked",
        "/home/vishu/reports/rat-mapping/circ/hb/4000w_4000s_4c_4t_15d_95a_tm/vertices/hb_mvn_rl_masked",
        "/home/vishu/reports/rat-mapping/circ/hb/4000w_4000s_4c_4t_15d_95a_tm/vertices/robust_hb_mvn_rl_masked",
    ]
    out += [threshold_analysis(model_dir, correction=True, dump=True)[0][0] for model_dir in dirs1]

    dirs2 = [
        "/home/vishu/reports/rat-mapping/circ/estimation/4000w_4000s_4c_4t_15d_95a_tm/diam/circ_est_mvn_reference_rl_masked",
        "/home/vishu/reports/rat-mapping/circ/estimation/4000w_4000s_4c_4t_15d_95a_tm/diam/robust_circ_est_mvn_reference_rl_masked",
        "/home/vishu/reports/rat-mapping/circ/estimation/4000w_4000s_4c_4t_15d_95a_tm/radii/circ_est_mvn_reference_rl_masked",
        "/home/vishu/reports/rat-mapping/circ/estimation/4000w_4000s_4c_4t_15d_95a_tm/radii/robust_circ_est_mvn_reference_rl_masked",
        "/home/vishu/reports/rat-mapping/circ/estimation/4000w_4000s_4c_4t_15d_95a_tm/vertices/circ_est_mvn_reference_rl_masked",
        "/home/vishu/reports/rat-mapping/circ/estimation/4000w_4000s_4c_4t_15d_95a_tm/vertices/robust_circ_est_mvn_reference_rl_masked",
    ]
    out += [estimation_analysis(model_dir)[0][0] for model_dir in dirs2]

    # dirs3 = list(zip(dirs1, dirs2))
    # out += [figure(mdirs, correction=True)[0][0] for mdirs in dirs3]

    return out


if __name__ == "__main__":
    out = main()
    output_path = os.path.join(BUILD_DIR, "circ.pdf")
    make_pdf(out, output_path)
