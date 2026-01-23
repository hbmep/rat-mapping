# figures.py
import os

import matplotlib as mpl
os.environ.setdefault("MPLBACKEND", "Agg")
mpl.use("Agg", force=True)

import matplotlib.pyplot as plt

from core_circ import fit as fit_circ
from core_size import fit as fit_size
from util_circ import (
    plot_thresholds as plot_thresholds_circ,
    plot_model as plot_model_circ,
)
from util_size import (
    plot_thresholds as plot_thresholds_size,
    plot_model as plot_model_size,
)
from constants import BUILD_DIR
os.makedirs(BUILD_DIR, exist_ok=True)

from paper.constants import HOME
ICONS_DIR = os.path.join(HOME, "bits")
KWS = dict(
    kw_fs=10, kw_ls=10, kw_ylim=(16, 512),
    kw_yticks=(16, 32, 64, 128, 256, 512),

    # kw_fs=10, kw_ls=10, kw_ylim=(30, 400),
    # kw_yticks=(32, 64, 128, 256),
)
FG_KWS = dict(
    kw_mean_color="0.05", kw_mean_alpha=0.95,
    kw_mean_lw=1, kw_mean_marker="o", kw_mean_marker_size=18,
)
BG_KWS = dict(
    kw_rat_cmap="Greys",
    kw_rat_line_alpha=0.25,
    kw_rat_line_lw=1.1,
    kw_rat_point_color="0.55",
    kw_rat_point_alpha=0.25,
    kw_rat_point_size=14,
)


def plot_circ(
    df,
    *,
    result,
    rhs_terms,
    indicator_columns,
    reference_term,
    icons_dir,
):
    """
    Create one figure:
      - Top row: 3 subplots (diameters/radii/vertices) via plot_thresholds_circ
      - Bottom row: 1 subplot spanning all columns via plot_model_circ
    """
    plt.close("all")
    print("Creating figure grid ...")
    fig = plt.figure(figsize=(11, 7))
    ncols = 21
    num_plots = 3
    ratios = [4, 8, 9]
    width_ratios = []
    wspace = 0.4
    for r in ratios:
        width_ratios += [1] * r
        width_ratios.append(wspace)
    width_ratios = width_ratios[:-1]
    gs = fig.add_gridspec(
        nrows=2, ncols=ncols + (num_plots - 1),
        width_ratios=width_ratios,
        height_ratios=[1.0, 1.3],
        hspace=0.25,
        wspace=0.,
    )
    axes_top = []
    cum_sum = 0
    for r in ratios:
        axes_top.append(fig.add_subplot(gs[0, cum_sum:cum_sum + r]))
        cum_sum += r
        cum_sum += 1
    k = 1
    # ax_bottom = fig.add_subplot(gs[1, k:-k])
    ax_bottom = fig.add_subplot(gs[1, :])
    fig.show()

    kws = {u: v for u, v in KWS.items()}
    fg_kws = {u: v for u, v in FG_KWS.items()}
    bg_kws = {u: v for u, v in BG_KWS.items()}
    fs = 10
    kws['kw_fs'] = fs
    kws['kw_ls'] = fs
    icons_params = (0.09, -0.06)    # (zoom, y-offset)
    plot_thresholds_circ(
        df,
        axes=axes_top,
        result=result,
        remove_random_intercept=True,
        icons_dir=icons_dir,
        icons_params=icons_params,
        color_by_rat=True,
        show_mean=True,
        **kws,
        **fg_kws,
        **bg_kws
    )

    icons_params = (0.1, -0.06)    # (zoom, y-offset)
    reference_yoffset = -0.21
    plot_model_circ(
        result,
        ax=ax_bottom,
        rhs_terms=rhs_terms,
        indicator_terms=indicator_columns,
        reference_term=reference_term,
        icons_dir=icons_dir,
        icons_params=icons_params,
        as_percent=True,
        reference_yoffset=reference_yoffset,
        **kws
    )
    # (left, bottom, right, top)
    adjust = (0.075, 0.13, 0.99, 0.98)
    fig.subplots_adjust(*adjust)
    fig.align_labels()
    fig.align_xlabels()
    fig.align_ylabels()
    return fig


def main_circ():
    (model,
    result,
    indicator_columns,
    rhs_terms,
    set_reference,
    formula,
    df) = fit_circ()
    fig = plot_circ(
        df,
        result=result,
        rhs_terms=rhs_terms,
        indicator_columns=indicator_columns,
        reference_term=set_reference,
        icons_dir=ICONS_DIR
    )
    out = os.path.join(BUILD_DIR, "circ.svg")
    fig.savefig(out)
    print(f"Saved to {out}")
    out = os.path.join(BUILD_DIR, "circ.png")
    fig.savefig(out, dpi=600)
    print(f"Saved to {out}")
    return


def plot_size(
    df,
    *,
    result,
    rhs_terms,
    indicator_columns,
    reference_term,
    icons_dir,
):
    plt.close("all")
    print("Creating figure grid ...")
    fig = plt.figure(figsize=(11, 7))
    ncols = 21
    num_plots = 3
    ratios = [5, 10, 6]
    width_ratios = []
    wspace = 0.4
    for r in ratios:
        width_ratios += [1] * r
        width_ratios.append(wspace)
    width_ratios = width_ratios[:-1]
    gs = fig.add_gridspec(
        nrows=2, ncols=ncols + (num_plots - 1),
        width_ratios=width_ratios,
        height_ratios=[1.0, 1.3],
        hspace=0.2,
        wspace=0.,
    )
    axes_top = []
    cum_sum = 0
    for r in ratios:
        axes_top.append(fig.add_subplot(gs[0, cum_sum:cum_sum + r]))
        cum_sum += r
        cum_sum += 1
    k = 6
    ax_bottom = fig.add_subplot(gs[1, k:-k])
    fig.show()

    kws = {u: v for u, v in KWS.items()}
    fg_kws = {u: v for u, v in FG_KWS.items()}
    bg_kws = {u: v for u, v in BG_KWS.items()}
    fs = 9; kws['kw_fs'] = fs; kws['kw_ls'] = fs
    aa = 0.045
    bb = -0.06
    dd = 0
    icons_params=[
        (aa, bb, dd, 1),
        (aa, bb, dd, -1),
        (aa, bb, dd, -1),
    ]
    plot_thresholds_size(
        df,
        axes=axes_top,
        result=result,
        remove_random_intercept=True,
        icons_dir=icons_dir,
        icons_params=icons_params,
        color_by_rat=True,
        show_mean=True,
        **kws,
        **fg_kws,
        **bg_kws
    )
    icons_params = (0.07, -0.05)    # (zoom, y-offset)
    reference_yoffset = -0.1
    plot_model_size(
        result,
        ax=ax_bottom,
        rhs_terms=rhs_terms,
        indicator_terms=indicator_columns,
        reference_term=reference_term,
        icons_dir=icons_dir,
        icons_params=icons_params,
        as_percent=True,
        reference_yoffset=reference_yoffset,
        **kws
    )
    # (left, bottom, right, top)
    adjust = (0.065, 0.065, 0.99, 0.98)
    fig.subplots_adjust(*adjust)
    fig.align_labels()
    fig.align_xlabels()
    fig.align_ylabels()
    return fig


def main_size():
    (model,
    result,
    indicator_columns,
    rhs_terms,
    set_reference,
    formula,
    df,
    electrode_size,
    include_interaction) = fit_size()
    assert not include_interaction
    fig = plot_size(
        df,
        result=result,
        rhs_terms=rhs_terms,
        indicator_columns=indicator_columns,
        reference_term=set_reference,
        icons_dir=ICONS_DIR
    )
    out = os.path.join(BUILD_DIR, "size.svg")
    fig.savefig(out)
    print(f"Saved to {out}")
    out = os.path.join(BUILD_DIR, "size.png")
    fig.savefig(out, dpi=600)
    print(f"Saved to {out}")
    return


def main():
    main_circ()
    main_size()
    return


if __name__ == "__main__":
    main()
