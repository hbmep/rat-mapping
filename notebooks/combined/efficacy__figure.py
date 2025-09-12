import os
import pickle
import warnings

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from hbmep.util import site

from paper.util import (
    clear_axes,
    make_pdf,
    load_dump,
)
from efficacy__analysis import arrange, BUILD_DIR as EFFICACY_DIR, MODEL_DIR
from core__hb import BUILD_DIR

BUILD_DIR = os.path.join(BUILD_DIR, "efficacy__figure")
os.makedirs(BUILD_DIR, exist_ok=True)

plt.rcParams["svg.fonttype"] = "none"
LABEL_SIZE = 10
ROTATION = 35


def plot_efficacy_2p(run_id, load_dir):
    src = os.path.join(load_dir, f"{run_id}.pkl")
    positions, a, = load_dump(src)
    a = a[..., 0, :] - a[..., 1, :]
    a *= -1     # S - B
    a = np.mean(a, axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # # a = a[:, :, 1:2, ...]
        # a = np.where(
        #     np.isnan(a[:, :, 1, ...]),
        #     a[:, :, 0, ...],
        #     a[:, :, 1, ...]
        # )
        # a = a[:, :, None, ...]

        a = np.nanmean(a, axis=-2)

    diff_mean = np.nanmean(a, axis=(0, 1))
    positions, _ = zip(*sorted(
        zip(positions, diff_mean),
        key=lambda x: x[1]
    ))
    colors = sns.color_palette(palette="viridis", n_colors=len(positions))
    _, t = zip(*positions)
    colors = dict(zip(t, colors))

    figsize = (8, 5)
    width_ratios=[.35, .65]
    nr, nc = 1, 2
    fig, axes = plt.subplots(
        *(nr, nc), figsize=figsize, squeeze=False, constrained_layout=True,
        width_ratios=width_ratios
    )

    ax = axes[0, 0]
    y = list(range(8))[::-1]
    for position_idx, position in positions:
        color = colors[position]
        x = a[0, :, position_idx]
        ax.plot(x, y, color=color, label=position)
        sns.scatterplot(x=x, y=y, color=color, ax=ax)
    ax.set_yticks(y)
    ax.axvline(x=0, linestyle="--", color="k")
    ax.set_xlabel(
        "% Threshold change of Small from Large electrodes " + r"$(\log_2)$"
        + "\n" + r"($\rightarrow$ Large electrodes are more effective)",
        fontsize=LABEL_SIZE
    )
    ax.legend(reverse=True)
    y = [f"rat0{u + 1}" for u in range(8)]
    ax.set_yticklabels(y, size="medium")
    x = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    lx = [np.log2(u / 100) for u in x]
    lx = [np.round(u, 2) for u in lx]
    ticklabels = [f"{v - 100}%" for i, (u, v) in enumerate(zip(lx, x))]
    ax.set_xticks(lx)
    ax.set_xticklabels(ticklabels, rotation=ROTATION)
    ax.set_xlim(right=1.6, left=-1.035)

    diff_mean = np.nanmean(a, axis=(0, 1))
    diff_err = stats.sem(a, axis=(0, 1), nan_policy="omit")
    ax = axes[0, 1]
    for position_idx, position in positions:
        color = colors[position]
        ax.barh(
            position,
            diff_mean[position_idx],
            color=color,
            xerr=diff_err[position_idx],
            capsize=5
        )
    ax.set_xlabel(
        "% Threshold increase of Small from Large electrodes " + r"$(\log_2)$"
        + "\n" + r"($\rightarrow$ Large electrodes are more effective)",
        fontsize=LABEL_SIZE
    )
    x = [100, 125, 150, 175, 200, 225, 250, 275, 300]
    lx = [np.log2(u / 100) for u in x]
    lx = [np.round(u, 2) for u in lx]
    ticklabels = [f"{v - 100}%" for u, v in zip(lx, x)]
    ax.set_xticks(lx)
    ax.set_xticklabels(ticklabels, rotation=ROTATION)
    ax.set_xlim(right=1.6)

    test = stats.ttest_1samp(
        a[0], axis=0, nan_policy="omit", popmean=0
    )
    from statsmodels.stats.multitest import multipletests
    corrected = multipletests(test.pvalue, method="holm")[1]
    pvalue = []
    for position_idx, position in positions:
        pvalue.append((position, np.round(corrected[position_idx], 3)))
    # fig.suptitle(f"corrected: {pvalue}")

    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            ax.spines[['top', 'right']].set_visible(False)
    fig.align_xlabels()
    fig.align_ylabels()
    return (fig, axes), 


def plot(
    run_id,
    model_dir,
    load_dir,
    consistent_colors=False,
):
    if run_id == "size-ground":
        return plot_efficacy_2p(run_id, load_dir)

    df, model, posterior, subjects, positions, = arrange(run_id, model_dir)
    src = os.path.join(load_dir, f"{run_id}.pkl")
    (
        diff_positions, measure_mean, diff, diff_mean, diff_err, negate,
    ) = load_dump(src)

    if run_id == "shie":
        diff_positions = [
            (u, (
                v.replace("__", " ")
                .replace("Pseudo-Mono", "(PS)")
                .replace("Biphasic", "(B)")
            ))   
            for u, v in diff_positions
        ]
        positions = [
            (u, (
                v.replace("__", " ")
                .replace("Pseudo-Mono", "(PS)")
                .replace("Biphasic", "(B)")
            ))   
            for u, v in positions
        ]

    colors = sns.color_palette(palette="viridis", n_colors=len(positions))
    _, t = zip(*diff_positions)
    colors = dict(zip(t, colors))
    if consistent_colors:
        colors = sns.color_palette(palette="tab10", n_colors=len(positions))
        colors = dict(zip(sorted(t), colors))

    figsize = (8, 5)
    width_ratios=[.35, .65]
    nr, nc = 1, 2
    fig, axes = plt.subplots(
        *(nr, nc), figsize=figsize, squeeze=False, constrained_layout=True,
        width_ratios=width_ratios
    )

    ax = axes[0, 0]
    y = list(range(len(subjects)))[::-1]
    for position_idx, position in diff_positions:
        color = colors[position]
        x = measure_mean[:, position_idx]
        ax.plot(x, y, color=color, label=position)
        sns.scatterplot(x=x, y=y, color=color, ax=ax)
    ax.set_yticks(y)
    _, y = zip(*subjects)
    y = [u.replace("amap", "rat") for u in y]
    ax.set_yticklabels(y, size="medium")
    ax.set_xlabel(
        "Average threshold across muscles " + r"$(\mu$A, $\log_2$)" + "\n"
        + r"($\leftarrow$ lower is more effective)",
        fontsize=LABEL_SIZE
    )
    ax.set_xticks([4, 6, 8])
    ax.set_xlim(left=3.6, right=8.9)

    ax = axes[0, 1]
    for pos_idx, pos_inv in diff_positions:
        xme = diff_mean[:, -1]
        xerr = diff_err[..., :, -1]
        xme = xme[pos_idx]
        xerr = xerr[pos_idx]
        ax.errorbar(
            x=xme,
            xerr=xerr,
            y=pos_inv,
            fmt="o",
            ecolor=colors[pos_inv],
            color=colors[pos_inv],
        )
    ax.vlines(
        xme,
        linestyle="--",
        color="k",
        ymax=(len(positions) - 1),
        ymin=0,
        zorder=-200
    )
    ax.tick_params(axis="y", rotation=0)
    ax.set_xlabel(
        "% Threshold increase " + r"$(\log_2)$" + "\n"
        + r"($\leftarrow$ lower is more effective)",
        fontsize=LABEL_SIZE
    )
    if run_id in {"diam", "vertices", "shie", "lat-big-ground"}:
        ax.set_xlim(right=1.5)
        ax.set_xticks([0, .6, 1.2])
    elif run_id in {"radii"}:
        ax.set_xlim(right=1.)
        ax.set_xticks([0, .4, .8])
    elif run_id in {"lat-small-ground"}:
        ax.set_xlim(right=1.2)
        ax.set_xticks([0, .5, 1.])

    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            if ax.get_legend(): ax.get_legend().remove()
            sides = ["top", "right"]
            ax.spines[sides].set_visible(False)

    ax = axes[0, 1]
    legend_loc = "upper right"
    ax.legend(
        *axes[0, 0].get_legend_handles_labels(), loc=legend_loc,
        frameon=True, reverse=True,
    )
    fig.align_xlabels()
    fig.align_ylabels()
    return (fig, axes),


def efficacy(model_dir, consistent_colors=False):
    run_ids = [
        "diam", "radii", "vertices",
        "shie",
        "lat-small-ground", "lat-big-ground",
        "size-ground"
    ]
    out = []
    axs = []
    for run_id in run_ids:
        fig, = plot(
            run_id, model_dir, EFFICACY_DIR,
            consistent_colors=consistent_colors
        )
        fig, axes = fig
        if run_id != "size-ground":
            axs.append(axes)
            axes[0][1].tick_params(axis="x", rotation=ROTATION)
        output_path = os.path.join(BUILD_DIR, f"{run_id}.svg")
        out.append((fig, output_path))

    if axs:
        [axes[0, 0].sharey(axs[0][0, 0]) for axes in axs]
        for idx in [[0, 0], [0, 1]]:
            x = [axes[*idx].get_xlim() for axes in axs]
            left = min([u for u, v in x])
            right = max([v for u, v in x])
            [axes[*idx].sharex(axs[0][*idx]) for axes in axs]
            axs[0][*idx].set_xlim(left, right)
        
        x = [100, 125, 150, 175, 200, 225, 250, 275, 300]
        lx = [np.log2(u / 100) for u in x]
        lx = [np.round(u, 2) for u in lx]
        ticklabels = [f"{v - 100}%" for u, v in zip(lx, x)]
        axs[0][0, 1].set_xticks(lx)
        axs[0][0, 1].set_xticklabels(ticklabels)
        axs[0][0, 1].set_xlim(right=1.6)

        ticks = [4, 5, 6, 7, 8]
        axs[0][0, 0].set_xticks(ticks)
        axs[0][0, 0].set_xticklabels([2 ** i for i in ticks])

    [fig.savefig(output_path, dpi=600) for fig, output_path in out]
    output_path = os.path.join(BUILD_DIR, "efficacy.pdf")
    make_pdf([fig for fig, _ in out], output_path)
    return


def main():
    model_dir = MODEL_DIR
    CONSISTENT_COLORS = False
    efficacy(model_dir, consistent_colors=CONSISTENT_COLORS)
    return


if __name__ == "__main__":
    main()
