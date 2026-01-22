# figures.py
import os

import matplotlib as mpl
os.environ.setdefault("MPLBACKEND", "Agg")
mpl.use("Agg", force=True)

import matplotlib.pyplot as plt

from core_circ import fit as fit_circ
from util import plot_thresholds, plot_model
from constants import BUILD_DIR
os.makedirs(BUILD_DIR, exist_ok=True)


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
      - Top row: 3 subplots (diameters/radii/vertices) via plot_thresholds
      - Bottom row: 1 subplot spanning all columns via plot_model
    """
    print("Creating figure grid ...")
    figsize=(11, 7)
    fig = plt.figure(
        figsize=figsize,
        # constrained_layout=True
    )
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        width_ratios=[4, 8, 9],
        height_ratios=[1.0, 1.3],
        hspace=0.3,
        wspace=0.08,
    )
    axes_top = [fig.add_subplot(gs[0, j]) for j in range(3)]
    ax_bottom = fig.add_subplot(gs[1, :])

    icons_dir = "/home/vishu/bits"
    icons_params = (0.12, -0.06)    # (zoom, y-offset)
    kws = dict(
        kw_fs=10, kw_ls=10, kw_ylim=(16, 512),
        kw_yticks=(16, 32, 64, 128, 256, 512),
    )
    fg_kws = dict(
        kw_mean_color="0.05", kw_mean_alpha=0.95,
        kw_mean_lw=1, kw_mean_marker="o", kw_mean_marker_size=18,
    )
    bg_kws = dict(
        kw_rat_cmap="Greys",
        kw_rat_line_alpha=0.25,
        kw_rat_line_lw=1.1,
        kw_rat_point_color="0.55",
        kw_rat_point_alpha=0.25,
        kw_rat_point_size=14,
    )
    plot_thresholds(
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

    reference_yoffset = -0.25
    plot_model(
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


def main():
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
        icons_dir="/home/vishu/bits",
    )
    out = os.path.join(BUILD_DIR, "circ.svg")
    fig.savefig(out)
    print(f"Saved to {out}")
    out = os.path.join(BUILD_DIR, "circ.png")
    fig.savefig(out, dpi=600)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
