# util.py
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

SEM_TIMES = 1
NEG_COLOR = "#4C78A8"


def add_icons(
    ax,
    x_positions,
    labels,
    icons_dir,
    icons_params,   # (zoom, y-offset)
    **kws,
):
    reference_term = kws.get("reference_term")
    reference_yoffset = kws.get("reference_yoffset")
    kw_fs = kws.get("kw_fs")
    kw_ls = kws.get("kw_ls")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([""] * len(labels))
    for xi, lab in zip(x_positions, labels):
        fname = lab
        if fname.startswith("-"):
            fname = fname[1:]
        src = os.path.join(icons_dir, f"{fname}.png")
        img = mpl.image.imread(src)
        imagebox = mpl.offsetbox.OffsetImage(img, zoom=icons_params[0])
        ab = mpl.offsetbox.AnnotationBbox(
            imagebox,
            (xi, icons_params[1]),
            xycoords=ax.get_xaxis_transform(),
            frameon=False,
            box_alignment=(0.5, 1.0),
            pad=0.0,
        )
        ax.add_artist(ab)
        if reference_term is not None and lab == reference_term:
            reference_text_kwargs = dict(
                fontsize=kw_fs,
                ha="center",
                va="top",
            )
            ax.text(
                xi, reference_yoffset,
                "Baseline",
                transform=ax.get_xaxis_transform(),
                **reference_text_kwargs,
            )
    return


def plot_thresholds(
    df,
    *,
    axes,
    result,
    remove_random_intercept=True,
    icons_dir=None,
    icons_params=(0.18, -0.12, 0.28),
    color_by_rat=True,
    show_mean=True,
    **kws
):
    kw_fs = kws.get("kw_fs")
    kw_ls = kws.get("kw_ls")
    kw_ylim = kws.get("kw_ylim", (16, 512))
    kw_yticks = kws.get("kw_yticks", (16, 32, 64, 128, 256, 512))

    kw_rat_cmap = kws.get("kw_rat_cmap", "Greys")
    kw_rat_line_alpha = kws.get("kw_rat_line_alpha", 0.25)
    kw_rat_line_lw = kws.get("kw_rat_line_lw", 1.1)
    kw_rat_point_color = kws.get("kw_rat_point_color", "0.55")
    kw_rat_point_alpha = kws.get("kw_rat_point_alpha", 0.25)
    kw_rat_point_size = kws.get("kw_rat_point_size", 14)

    kw_mean_color = kws.get("kw_mean_color", "0.05")
    kw_mean_alpha = kws.get("kw_mean_alpha", 0.95)
    kw_mean_lw = kws.get("kw_mean_lw", 2.6)
    kw_mean_marker = kws.get("kw_mean_marker", "o")
    kw_mean_marker_size = kws.get("kw_mean_marker_size", 18)

    diameters = ["SE-NW", "S-N", "NE-SW", "E-W"]
    radii = ["SE-C", "S-C", "SW-C", "W-C", "NW-C", "N-C", "NE-C", "E-C"]
    vertices = ["-C", "-SE", "-S", "-SW", "-W", "-NW", "-N", "-NE", "-E"]
    groups = [diameters, radii, vertices]

    rats = sorted(df.rat.unique())
    if color_by_rat:
        cmap = plt.get_cmap(kw_rat_cmap)
        ts = np.linspace(0.20, 0.85, len(rats))
        rat_color = {r: cmap(t) for r, t in zip(rats, ts)}
    else:
        rat_color = {r: "0.70" for r in rats}

    for ax, conds in zip(axes, groups):
        ax.clear()
        x = np.arange(len(conds), dtype=float)
        Y = []
        for ri, rat in enumerate(rats):
            idx = df.rat == rat
            sub = df.loc[idx].reset_index(drop=True)
            assert sub.shape[0] == 21
            mapping = dict(zip(sub.label, sub.a))
            y = []
            for ci, c in enumerate(conds):
                y.append(mapping[c])
            y = np.array(y)
            if remove_random_intercept:
                b_i = result.random_effects.get(rat, None).item()
                y = y - b_i
            ax.plot(
                x, y,
                color=rat_color[rat],
                alpha=kw_rat_line_alpha,
                linewidth=kw_rat_line_lw,
                zorder=1,
                label=f"rat{int(rat)+1:02d}",
            )
            ax.scatter(
                x, y,
                color=kw_rat_point_color,
                alpha=kw_rat_point_alpha,
                s=kw_rat_point_size,
                linewidths=0,
                zorder=2,
            )
            Y.append(y)
        Y = np.array(Y)
        if show_mean:
            y = np.nanmean(Y, axis=0)
            ax.plot(
                x, y,
                color=kw_mean_color,
                alpha=kw_mean_alpha,
                linewidth=kw_mean_lw,
                marker=kw_mean_marker,
                markersize=np.sqrt(kw_mean_marker_size),
                zorder=10,
                label="mean",
            )
        add_icons(
            ax,
            x_positions=x,
            labels=conds,
            icons_dir=icons_dir,
            icons_params=icons_params,
        )

    ticks = np.array(list(kw_yticks), dtype=float)
    ticks_log2 = np.log2(ticks)
    for j, ax in enumerate(axes):
        ax.set_yticks(ticks_log2)
        ax.set_yticklabels([f"{int(t)}" for t in ticks])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, alpha=0.2)
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(
            axis="both", left=True,
            labelleft=True if not j else False,
            # labelleft=True,
            bottom=True, labelbottom=False,
            labelsize=kw_ls
        )
        ax.sharey(axes[0])

    ax = axes[0]
    ax.set_ylabel(
        "Threshold ($\\mu$A, $\\log_2$ scale)"
        "\n(← lower is more effective)",
        fontsize=kw_fs
    )
    if kw_ylim is not None:
        lo, hi = float(kw_ylim[0]), float(kw_ylim[1])
        ax.set_ylim(np.log2(lo), np.log2(hi))

    return


def plot_model(
    result,
    *,
    ax,
    rhs_terms,
    indicator_terms,
    reference_term,
    icons_dir,
    icons_params,   # (zoom, y-offset)
    as_percent=True,
    reference_yoffset=-0.21,
    # # inset controls
    # add_distance_inset: bool = False,
    # df: Optional[pd.DataFrame] = None,
    # distance_term: str = "P",
    # distance_col: Optional[str] = None,
    # inset_bbox: Tuple[float, float, float, float] = (0.63, 0.08, 0.34, 0.34),
    # inset_loc: str = "lower right",
    # inset_point_size: float = 10.0,
    # inset_point_alpha: float = 0.55,
    # inset_line_lw: float = 1.5,
    **kws
):
    ax.clear()
    kw_fs = kws.get("kw_fs")
    kw_ls = kws.get("kw_ls")
    # kw_ylim = kws.get("kw_ylim", (16, 512))
    kw_yticks = (-60, -40, -20, 0, 20, 40, 60)

    assert len(set(indicator_terms)) == len(indicator_terms)
    params = result.params
    bse = result.bse

    rhs_no_int = [t for t in rhs_terms if t != "Intercept"]
    non_id_terms = [t for t in rhs_no_int if t not in indicator_terms]
    terms = list(indicator_terms) + non_id_terms
    x = np.arange(len(terms), dtype=float)

    beta = np.full(len(terms), np.nan, dtype=float)
    se = np.full(len(terms), np.nan, dtype=float)

    for k, t in enumerate(terms):
        if reference_term is not None and t == reference_term:
            beta[k] = 0.0
            se[k] = np.nan
            continue
        if t not in params.index:
            raise KeyError(
                f"Term {t!r} not in result.params. "
                f"If this is the reference term, pass reference_term={t!r}."
            )
        beta[k] = params[t]
        se[k] = bse[t]

    if as_percent:
        y = (np.power(2.0, beta) - 1.0) * 100.0
        yerr = None
        mask_err = np.isfinite(se)
        ylabel = (
            "% Threshold change vs baseline"
            "\n(← lower is more effective)"
        )


        if mask_err.any():
            k = SEM_TIMES
            beta_m = beta[mask_err]
            se_m = se[mask_err]
            y_m = y[mask_err]

            lo_beta = beta_m - k * se_m
            hi_beta = beta_m + k * se_m

            lo = (np.power(2.0, lo_beta) - 1.0) * 100.0
            hi = (np.power(2.0, hi_beta) - 1.0) * 100.0

            mean_minus_lo = y_m - lo
            hi_minus_mean = hi - y_m
            yerr = np.vstack([mean_minus_lo, hi_minus_mean])

    else:
        raise NotImplementedError

    neg_color = NEG_COLOR
    pos_color = "#E45756"
    pos_color = neg_color
    colors = [neg_color if v < 0 else pos_color for v in y]

    ax.bar(
        x, y,
        color=colors,
        edgecolor="0.2",
        linewidth=0.9,
        zorder=1000,
    )

    if mask_err.any() and yerr is not None:
        ax.errorbar(
            x[mask_err],
            y[mask_err],
            yerr=yerr,
            fmt="none",
            ecolor="k",
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
            zorder=10_000,
        )

    add_icons(
        ax,
        x_positions=x[:-len(non_id_terms)],
        labels=terms[:-len(non_id_terms)],
        icons_dir=icons_dir,
        icons_params=icons_params,
        reference_term=reference_term,
        reference_yoffset=reference_yoffset,
        kw_fs=kw_fs
    )

    mapping = {
        "P": r"$\frac{1}{\mathrm{Distance}}$",
        "OC": r"$\mathrm{cosine}$",
        "OS": r"$\mathrm{sine}$"
    }
    mapping = {
        "P": "Distance\npenalty",
        # "OC": "Cosine\ncomponent",
        # "OS": "Sine\ncomponent"
        "OC": "Cosine",
        "OS": "Sine"
    }
    # mapping = {
    #     "P": r"$\displaystyle \frac{1}{\mathrm{Distance}}$",
    #     "OC": "cosine",
    #     "OS": "sine"
    # }

    labels = [mapping[u] if u in mapping else "" for u in terms]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=kw_fs)

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yticks(kw_yticks)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=kw_fs)
    ax.tick_params(
        axis="y", left=True, labelleft=True,
        # bottom=True, labelbottom=False,
        labelsize=kw_ls
    )

    return
