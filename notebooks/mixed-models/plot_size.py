import os

import matplotlib as mpl
os.environ.setdefault("MPLBACKEND", "Agg")
mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"

from util_size import (
    plot_thresholds,
    plot_model,
)
from core_size import fit
from core_circ import BUILD_DIR
from plot_circ import KWS, BG_KWS, FG_KWS, BUILD_DIR


def main():
    result, indicator_columns = fit()
    fig = plot(result, indicator_terms=indicator_columns)

    out = os.path.join(BUILD_DIR, "mixed-size.png")
    fig.savefig(out, dpi=600)
    print(f"Saved to {out}")
    out = os.path.join(BUILD_DIR, "mixed-size.svg")
    fig.savefig(out, dpi=1200)
    out = os.path.join(BUILD_DIR, "mixed-size.pdf")
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
    ratios = [5, 10, 6]
    width_ratios = []
    wspace = .5
    for r in ratios:
        width_ratios += [1] * r
        width_ratios.append(wspace)
    width_ratios = width_ratios[:-1]
    gs = fig.add_gridspec(
        nrows=2, ncols=ncols + (num_plots - 1),
        width_ratios=width_ratios,
        height_ratios=[1, 1.2],
        hspace=0.2,
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
    fs = 9; kws['kw_fs'] = fs; kws['kw_ls'] = fs
    # aa = 0.04
    aa = 0.038
    bb = -0.05
    dd = 0
    icons_params=[
        (aa, bb, dd, 1),
        (aa, bb, dd, -1),
        (aa, bb, dd, -1),
    ]
    plot_thresholds(
        df,
        axes=axes_top,
        result=result,
        icons_params=icons_params,
        **kws,
        **fg_kws,
        **bg_kws
    )

    icons_params = (0.06, -0.05)    # (zoom, y-offset)
    reference_yoffset = -0.19
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
        **kws
    )

    # (left, bottom, right, top)
    adjust = (0.065, 0.11, 0.99, 0.98)
    fig.subplots_adjust(*adjust)

    fig.align_labels()
    fig.align_xlabels()
    fig.align_ylabels()

    return fig


if __name__ == "__main__":
    main()
