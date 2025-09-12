import os
import pickle
import inspect

import pandas as pd
from jax import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

import hbmep as mep
from hbmep import functional as F
from hbmep.util import site

from paper.analysis import load_smalar as load
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

BUILD_DIR = "/home/vishu/reports/rat-mapping/combined/efficacy__figure"
os.makedirs(BUILD_DIR, exist_ok=True)

plt.rcParams["svg.fonttype"] = "none"
LABEL_SIZE = 10


def estimation_lat(model_dir, fig, ax):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
		positions,
        degrees,
        sizes,
        num_features,
        mask_features,
        *_,
    ) = load(model_dir, estimation=True)
    positions = degrees.copy()
    palette = "viridis"
    colors = sns.color_palette(palette=palette, n_colors=len(positions))

    param = posterior["a_delta_loc"]
    param_mean = [0] + np.mean(param, axis=0).tolist()
    _, positions = map(list, zip(*sorted(
        zip(param_mean, positions),
        key=lambda x: (-x[0], x[1][0])
    )))

    for i, (idx, position) in enumerate(positions):
        color = colors[i]
        if not idx:
            label = position
            label += " (reference)"
            label = label[1:]
            ax.axvline(x=0, label=label, color=color, linestyle="--", ymax=.95)
            continue
        samples = param[:, idx - 1]
        sns.kdeplot(samples, color=color, label=position, ax=ax, bw_adjust=1.4)

    ax.legend(reverse=True, title="Order (most effective to least)")
    return


def main_lat():

    model_dirs = [

        # "/home/vishu/reports/rat-mapping/smalar/estimation/4000w_4000s_4c_4t_15d_95a_tm/lat-small-ground/lat_est_mvn_block_reference_rl_masked",
        # "/home/vishu/reports/rat-mapping/smalar/estimation/4000w_4000s_4c_4t_15d_95a_tm/lat-big-ground/lat_est_mvn_block_reference_rl_masked",
        "/home/vishu/reports/rat-mapping/smalar/estimation/4000w_4000s_4c_4t_15d_95a_tm/lat-small-ground/robust_lat_est_mvn_block_reference_rl_masked",
        "/home/vishu/reports/rat-mapping/smalar/estimation/4000w_4000s_4c_4t_15d_95a_tm/lat-big-ground/robust_lat_est_mvn_block_reference_rl_masked",

    ]

    nr, nc = 1, 2
    fig, axes = plt.subplots(
        *(nr, nc), figsize=(9, 5), sharey=True, constrained_layout=True,
        squeeze=False, sharex=True
    )

    for i, model_dir in enumerate(model_dirs):
        estimation_lat(model_dir, fig, axes[0, i])

    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.tick_params(axis="both", left=False, labelleft=False)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_ylim(bottom=-.02)
            # ax.xaxis.set_major_locator(MaxNLocator(5))
    
    x = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350]
    lx = [np.log2(u / 100) for u in x]
    lx = [np.round(u, 2) for u in lx]
    ticklabels = [f"{v - 100}%" for u, v in zip(lx, x)]
    ax = axes[0, 0]
    ax.set_xticks(lx)
    ax.set_xticklabels(ticklabels, rotation=90)
    for j in range(nc):
        axes[0, j].tick_params(axis="x", labelrotation=35, labelsize=LABEL_SIZE)
   
    fig.supxlabel(
        "% Threshold change from reference" + r" $(\log_2)$" "\n"
        + r"($\leftarrow$ lower is more effective)",
        fontsize=LABEL_SIZE,
    )
    fig.align_xlabels()
    fig.align_ylabels()

    output_path = os.path.join(BUILD_DIR, f"estimation_lat.svg")
    fig.savefig(output_path, dpi=600)
    output_path = os.path.join(BUILD_DIR, "estimation_lat.pdf")
    make_pdf([fig], output_path)
    return


def estimation_size(model_dir, fig, ax):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
		positions,
        degrees,
        sizes,
        num_features,
        mask_features,
        *_,
    ) = load(model_dir, estimation=True)
    positions = degrees.copy()
    palette = "viridis"
    colors = sns.color_palette(palette=palette, n_colors=len(positions))

    param = posterior["a_delta_loc"][:, 0, ...]
    param_mean = np.mean(param, axis=0).tolist()
    _, positions = map(list, zip(*sorted(
        zip(param_mean, positions),
        key=lambda x: (-x[0], x[1][0])
    )))

    ax.axvline(x=0, color="k", linestyle="--", ymax=.95)
    for i, (idx, position) in enumerate(positions):
        color = colors[i]
        samples = param[:, idx]
        sns.kdeplot(-samples, color=color, label=position, ax=ax, bw_adjust=1.4)

    ax.legend(reverse=True, title="Order (most effective to least)")
    return


def main_size():

    model_dirs = [

        # "/home/vishu/reports/rat-mapping/smalar/estimation/4000w_4000s_4c_4t_15d_95a_tm/size-ground/size_est_mvn_block_reference_rl_masked",
        "/home/vishu/reports/rat-mapping/smalar/estimation/4000w_4000s_4c_4t_15d_95a_tm/size-ground/robust_size_est_mvn_block_reference_rl_masked"

    ]

    nr, nc = 1, 1
    fig, axes = plt.subplots(
        *(nr, nc), figsize=(5, 5), sharey=True, constrained_layout=True,
        squeeze=False, sharex=True
    )

    for i, model_dir in enumerate(model_dirs):
        estimation_size(model_dir, fig, axes[0, i])

    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.tick_params(axis="both", left=False, labelleft=False)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_ylim(bottom=-.01)
            # ax.xaxis.set_major_locator(MaxNLocator(5))
    
    x = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350]
    lx = [np.log2(u / 100) for u in x]
    lx = [np.round(u, 2) for u in lx]
    ticklabels = [f"{v - 100}%" for u, v in zip(lx, x)]
    ax = axes[0, 0]
    ax.set_xticks(lx)
    ax.set_xticklabels(ticklabels, rotation=90)
    for j in range(nc):
        axes[0, j].tick_params(axis="x", labelrotation=35, labelsize=LABEL_SIZE)
   
    fig.supxlabel(
        "% Threshold change of Small from Large electrodes " + r"$(\log_2)$" + "\n"
        + r"($\rightarrow$ Large electrodes are more effective)",
        fontsize=LABEL_SIZE
    )
    fig.align_xlabels()
    fig.align_ylabels()

    output_path = os.path.join(BUILD_DIR, f"estimation_size.svg")
    fig.savefig(output_path, dpi=600)
    output_path = os.path.join(BUILD_DIR, "estimation_size.pdf")
    make_pdf([fig], output_path)
    return


if __name__ == "__main__":
    main_lat()
    main_size()
