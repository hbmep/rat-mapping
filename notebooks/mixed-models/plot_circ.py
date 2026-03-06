import os

import matplotlib as mpl
os.environ.setdefault("MPLBACKEND", "Agg")
mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"

from util_circ import (
    plot_thresholds,
    plot_model,
)
from core_circ import fit
from core_circ import BUILD_DIR


KWS = dict(
    kw_fs=10, kw_ls=10,

    remove_random_intercept=False,
    kw_y_scale="log2",
    kw_ylim=(16, 1024),
    kw_yticks=(16, 32, 64, 128, 256, 512),

    # remove_random_intercept=True,
    # kw_y_scale="log2",
    # kw_ylim=(28, 512),
    # kw_yticks=(16, 32, 64, 128, 256),

    # remove_random_intercept=False,
    # kw_y_scale="linear",
    # kw_ylim=(0, 400),
    # kw_yticks=(0, 100, 200, 300),

    # remove_random_intercept=True,
    # kw_y_scale="linear",
    # kw_ylim=(0, 320),
    # kw_yticks=(0, 100, 200, 300),

    # kw_ylim=(0, 400),
    # kw_yticks=(0, 50, 100, 150, 200, 250, 300, 350),

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


def main():
    result, indicator_columns = fit()

    fig = plot(result, indicator_terms=indicator_columns)

    out = os.path.join(BUILD_DIR, "mixed-circ.png")
    fig.savefig(out, dpi=600)
    print(f"Saved to {out}")
    out = os.path.join(BUILD_DIR, "mixed-circ.svg")
    fig.savefig(out, dpi=1200)
    out = os.path.join(BUILD_DIR, "mixed-circ.pdf")
    fig.savefig(out, dpi=1200)

    return


def plot(result, *, indicator_terms):
    (
        model,
        result,
        rhs_terms,
        set_reference,
        formula,
        df,
    ) = result

    plt.close("all")
    print("Creating figure grid ...")
    fig = plt.figure(figsize=(9, 5))
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
        hspace=0.3,
        wspace=0.,
    )
    axes_top = []
    cum_sum = 0
    for r in ratios:
        axes_top.append(fig.add_subplot(gs[0, cum_sum:cum_sum + r]))
        cum_sum += r
        cum_sum += 1
    ax_bottom = fig.add_subplot(gs[1, :])
    fig.show()

    kws = {u: v for u, v in KWS.items()}
    fg_kws = {u: v for u, v in FG_KWS.items()}
    bg_kws = {u: v for u, v in BG_KWS.items()}
    fs = 9
    kws['kw_fs'] = fs
    kws['kw_ls'] = fs
    icons_params = (0.09, -0.06)    # (zoom, y-offset)
    plot_thresholds(
        df,
        axes=axes_top,
        result=result,
        icons_params=icons_params,
        **kws,
        **fg_kws,
        **bg_kws
    )

    icons_params = (0.1, -0.06)    # (zoom, y-offset)
    reference_yoffset = -0.25
    plot_model(
        ax=ax_bottom,
        model=model,
        result=result,
        rhs_terms=rhs_terms,
        set_reference=set_reference,
        formula=formula,
        df=df,
        indicator_terms=indicator_terms,
        icons_params=icons_params,
        reference_term=set_reference,
        reference_yoffset=reference_yoffset,
        # orientation_style="bars",
        # orientation_style="polar",
        # orientation_style="bars-polar",
        # orientation_style="bars-polar",
        orientation_style="bars-polar-both",
        **kws
    )

    # (left, bottom, right, top)
    adjust = (0.065, 0.13, 0.99, 0.98)
    fig.subplots_adjust(*adjust)
    fig.align_labels()
    fig.align_xlabels()
    fig.align_ylabels()

    return fig


if __name__ == "__main__":
    main()
