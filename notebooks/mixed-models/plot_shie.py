import os

import matplotlib as mpl
os.environ.setdefault("MPLBACKEND", "Agg")
mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"

from constants import DATA_DIR, RESPONSE
from util_shie import (
    plot_thresholds,
    plot_model
)
from core_shie import fit
from core_circ import BUILD_DIR
from plot_circ import KWS, BG_KWS, FG_KWS, BUILD_DIR

NPY_PATH = os.path.join(DATA_DIR, "shie.npy")
TOML_PATH = os.path.join(DATA_DIR, "labels.toml")


def main():
    result, indicator_columns = fit()

    fig = plot(result, indicator_terms=indicator_columns)

    out = os.path.join(BUILD_DIR, "mixed-shie.png")
    fig.savefig(out, dpi=600)
    print(f"Saved to {out}")
    out = os.path.join(BUILD_DIR, "mixed-shie.svg")
    fig.savefig(out, dpi=1200)
    out = os.path.join(BUILD_DIR, "mixed-shie.pdf")
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
    fig = plt.figure(figsize=(5, 7))
    ncols = 8
    num_plots = 1
    ratios = [8]
    width_ratios = []
    wspace = 0.4
    for r in ratios:
        width_ratios += [1] * r
        width_ratios.append(wspace)
    width_ratios = width_ratios[:-1]
    gs = fig.add_gridspec(
        nrows=2, ncols=ncols + (num_plots - 1),
        width_ratios=width_ratios,
        height_ratios=[1.0, 1.2],
        hspace=0.4,
        wspace=0.,
    )
    axes_top = []
    cum_sum = 0
    for r in ratios:
        axes_top.append(fig.add_subplot(gs[0, cum_sum:cum_sum + r]))
        cum_sum += r
        cum_sum += 1
    k = 1
    ax_bottom = fig.add_subplot(gs[1, :])
    fig.show()
    
    kws = {u: v for u, v in KWS.items()}
    fg_kws = {u: v for u, v in FG_KWS.items()}
    bg_kws = {u: v for u, v in BG_KWS.items()}
    fs = 10
    kws['kw_fs'] = fs
    kws['kw_ls'] = fs
    icons_params = (0.1, -0.06)
    plot_thresholds(
        df,
        axes=axes_top,
        result=result,
        icons_params=icons_params,
        **kws,
        **fg_kws,
        **bg_kws
    )

    icons_params = (0.1, -0.06)
    reference_yoffset = -0.3
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
    adjust = (0.12, 0.14, 0.99, 0.98)
    fig.subplots_adjust(*adjust)
    fig.align_labels()
    fig.align_xlabels()
    fig.align_ylabels()

    return fig


if __name__ == "__main__":
    result = main()
