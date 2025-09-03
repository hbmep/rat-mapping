import os

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from paper.util import make_pdf
from core__hb import BUILD_DIR
from efficacy__analysis import run, BUILD_DIR, MODEL_DIR
os.makedirs(BUILD_DIR, exist_ok=True)

plt.rcParams["svg.fonttype"] = "none"


def annotate_heatmap(
    ax,
    cmap_arr,
    annot_arr,
    annot_position,
    **kw
):
    n = annot_arr.shape[0]
    annot_colors = np.where(cmap_arr > .6, "k", "white")
    for y in range(n):
        for x in range(n):
            if x >= y: continue
            text = annot_arr[y, x].item()
            ax.text(
                x + annot_position[0],
                y + annot_position[1],
                text,
                color=annot_colors[y, x],
                **kw,
            )
    return


def make_test(a, positions):
    def body_reorder(a, positions):
        a = a[0]
        diff = a[..., None] - a[..., None, :]
        diff = np.nanmean(diff, axis=0)
        t = (diff < 0).sum(axis=-1)
        ordering = np.argsort(t)
        a = a[..., ordering]
        positions = np.array(positions)[ordering].tolist()
        return a, positions,

    def mask_upper(arr):
        n = arr.shape[0]
        arr[np.triu_indices(n)] = np.nan
        return arr


    a, positions, = body_reorder(a, positions)
    diff = a[..., None] - a[..., None, :]
    diff = -diff
    test = stats.ttest_1samp(
        diff, axis=0, nan_policy="omit", popmean=0
    )
    pvalue = test.pvalue
    pvalue = mask_upper(pvalue)
    idx = np.tril_indices_from(pvalue, k=-1)
    corrected = stats.false_discovery_control(pvalue[idx])
    pvalue[idx] = corrected
    me = np.nanmean(diff, axis=0)
    change = ((2 ** me) - 1) * 100
    change = np.round(change, 0)
    change = change.astype(int).astype(str)
    change = np.char.add(change, "%")
    me = np.round(me, 3)
    me = me.astype(str)
    sem = stats.sem(diff, axis=0, nan_policy="omit")
    sem = np.round(sem, 3)
    sem = sem.astype(str)
    ci = np.char.add(np.char.add(me, " ± "), sem)
    df = test.df.astype(str)
    return positions, change, ci, pvalue, df,


def make_heatmap(run_id, positions, change, ci, pvalue, df):
    nr, nc = 1, 1
    figsize=(5, 5)
    fontsize = 8
    if run_id in {"vertices", "radii", "shie"}:
        figsize=(7.5, 5)
        fontsize= 7
    elif run_id in {"circ"}:
        figsize=(8, 10)
        fontsize= 5
    fig, axes = plt.subplots(
        *(nr, nc), figsize=figsize, constrained_layout=True,
        squeeze=False
    )

    ax = axes[0, 0]
    _, labels = zip(*positions)
    sns.heatmap(
        pvalue, annot=False, ax=ax, cbar=False, vmin=0, vmax=1,
        xticklabels=labels, yticklabels=labels,
    )
    annot_arr = np.round(pvalue, 3).astype(str)
    annot_arr = np.where(
        pvalue < 0.001,
        np.char.add(annot_arr, "***"),
        annot_arr
    )
    annot_arr = np.where(
        (pvalue >= 0.001) & (pvalue < 0.01),
        np.char.add(annot_arr, "**"),
        annot_arr
    )
    annot_arr = np.where(
        (pvalue >= 0.01) & (pvalue < 0.05),
        np.char.add(annot_arr, "*"),
        annot_arr
    )
    annot_kws = {"ha": 'center', "va": 'center'}
    annotate_heatmap(
        ax=ax, cmap_arr=pvalue, annot_arr=annot_arr,
        annot_position=(0.5, .75), **annot_kws, fontsize=fontsize
    )
    annot_arr = ci
    annot_kws = {"ha": 'center', "va": 'center'}
    annotate_heatmap(
        ax=ax, cmap_arr=pvalue, annot_arr=annot_arr,
        annot_position=(0.5, .5), **annot_kws, fontsize=fontsize
    )
    annot_arr = change
    annot_kws = {"ha": 'center', "va": 'center'}
    annotate_heatmap(
        ax=ax, cmap_arr=pvalue, annot_arr=annot_arr,
        annot_position=(0.5, .25), **annot_kws, fontsize=fontsize
    )
    annot_arr = df
    annot_kws = {"ha": 'left', "va": 'bottom'}
    annotate_heatmap(
        ax=ax, cmap_arr=pvalue, annot_arr=annot_arr,
        annot_position=(0, 1), **annot_kws, fontsize=fontsize
    )
    ax.tick_params(axis="both", rotation=0)
    return (fig, axes),


def main():
    model_dir = MODEL_DIR
    out = []

    # ids = ["diam", "radii", "vertices", "shie"]
    # a_mean = []
    # positions = []
    # counter = 0
    # for run_id_ in ids:
    #     curr_a_mean, curr_postions = run(run_id_, model_dir, skip_figure=True)
    #     a_mean.append(curr_a_mean)
    #     curr_postions = [(u + counter, v) for u, v in curr_postions]
    #     positions += curr_postions
    #     counter += len(curr_postions)
    # a_mean = np.concatenate(a_mean, axis=-1)
    # positions, change, ci, pvalue, df, = make_test(a_mean, positions)
    # run_id = "circ"
    # fig = make_heatmap(
    #     run_id, positions, change, ci, pvalue, df
    # )
    # fig = fig[0][0]
    # output_path = os.path.join(BUILD_DIR, f"heat_{run_id}.svg")
    # fig.savefig(output_path, dpi=600)

    run_ids = [
        "diam",
        "radii",
        "vertices",
        "shie",
    ]
    for run_id in run_ids:
        curr_a_mean, curr_positions = run(run_id, model_dir, skip_figure=True)
        (
            curr_positions, curr_change, curr_ci, curr_pvalue, curr_df, 
        ) = make_test(curr_a_mean, curr_positions)
        # idx = [
        #     i for i, (u, v) in enumerate(positions)
        #     if v in [z for _, z in curr_positions]
        # ]
        # curr_positions = np.array(positions)[idx].tolist()
        # curr_change = change[idx][..., idx]
        # curr_ci = ci[idx][..., idx]
        # curr_pvalue = pvalue[idx][..., idx]
        # curr_df = df[idx][..., idx]
        fig = make_heatmap(
            run_id, curr_positions, curr_change, curr_ci, curr_pvalue, curr_df
        )
        fig = fig[0][0]
        output_path = os.path.join(BUILD_DIR, f"heat_{run_id}.svg")
        fig.savefig(output_path, dpi=600)
        out.append(fig)

    run_ids = [
        "lat-small-ground",
        "lat-big-ground",
    ]
    for run_id in run_ids:
        a_mean, positions = run(run_id, model_dir, skip_figure=True)
        positions, change, ci, pvalue, df, = make_test(a_mean, positions)
        fig = make_heatmap(run_id, positions, change, ci, pvalue, df)[0][0]
        output_path = os.path.join(BUILD_DIR, f"heat_{run_id}.svg")
        fig.savefig(output_path, dpi=600)
        out.append(fig)

    output_path = os.path.join(BUILD_DIR, "heat.pdf")
    make_pdf(out, output_path)
    return


if __name__ == "__main__":
    main()
