import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from util_circ import (
    NEG_COLOR, FS_K, SEM_TIMES, ICONS_DIR,
    YLO, LABEL_MEAN, LABEL_FIXED, FCOLORS
)

FS_K = .9
INFO_SEM = 1.96

is_cc = "is_cathode_center"
is_pm = "is_pseudo_mono"
is_bi = "is_bi_phasic"
is_sh = "is_shielding"


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

    ax.set_xticks(x_positions)
    ax.set_xticklabels([""] * len(labels))
    for xi, lab in zip(x_positions, labels):
        fname = lab
        fname = fname.replace("k", "-")
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
                fontsize=kw_fs * FS_K,
                ha="center",
                va="top",
            )
            ax.text(
                xi,
                reference_yoffset,
                "(Baseline)",
                transform=ax.get_xaxis_transform(),
                **reference_text_kwargs,
            )
    return


def _apply_threshold_yaxis(
    axes,
    *,
    y_scale: str,   # "log2" or "linear"
    kw_yticks,
    kw_ylim,
    kw_fs,
    kw_ls,
    ylabel_text,
):
    if y_scale not in ("log2", "linear"):
        raise ValueError(
            f"kw_y_scale must be 'log2' or 'linear', got {y_scale!r}"
        )

    ticks = np.array(list(kw_yticks), dtype=float)

    if y_scale == "log2":
        tick_pos = np.log2(ticks)
        tick_lab = [f"{int(t)}" for t in ticks]
    else:
        tick_pos = ticks
        tick_lab = [f"{int(t)}" if float(t).is_integer() else f"{t:g}" for t in ticks]

    for j, ax in enumerate(axes):
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lab)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, alpha=0.2)
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(
            axis="both",
            left=True,
            labelleft=True if not j else False,
            bottom=True,
            labelbottom=False,
            labelsize=kw_ls,
        )

    axes[0].set_ylabel(
        ylabel_text,
        fontsize=kw_fs,
        linespacing=1.5,
    )

    if kw_ylim is not None:
        lo, hi = float(kw_ylim[0]), float(kw_ylim[1])
        if y_scale == "log2":
            axes[0].set_ylim(np.log2(lo), np.log2(hi))
        else:
            axes[0].set_ylim(lo, hi)

    for ax in axes[1:]:
        ax.sharey(axes[0])

    return


def plot_thresholds(
    df,
    *,
    axes,
    result,
    remove_random_intercept=True,
    icons_params=(0.18, -0.12, 0.28),
    color_by_rat=True,
    show_mean=True,
    **kws,
):
    icons_dir = ICONS_DIR
    kw_fs = kws.get("kw_fs")
    kw_ls = kws.get("kw_ls")
    kw_ylim = kws.get("kw_ylim", (16, 512))
    kw_yticks = kws.get("kw_yticks", (16, 32, 64, 128, 256, 512))

    kw_y_scale = kws.get("kw_y_scale", "log2")  # "log2" or "linear"

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

    # diameters = ["SE-NW", "S-N", "NE-SW", "E-W"]
    # radii = ["SE-C", "S-C", "SW-C", "W-C", "NW-C", "N-C", "NE-C", "E-C"]
    # vertices = ["-C", "-SE", "-S", "-SW", "-W", "-NW", "-N", "-NE", "-E"]
    # groups = [diameters, radii, vertices]

    mono = [
        '-C_Bi', 'C-_Bi',
        '-C_PM', 'C-_PM',
    ]
    hd = [
        'X-C_Bi', 'C-X_Bi',
        'X-C_PM', 'C-X_PM',
    ]
    mono = [u.replace("-", "k") for u in mono]
    hd = [u.replace("-", "k") for u in hd]
    groups = [mono + hd]

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
        Y_no_intercept = []

        for rat in rats:
            sub = df.loc[df.rat == rat].reset_index(drop=True)
            assert sub.shape[0] == 8
            mapping = dict(zip(sub.label, sub.a))

            y = np.array([mapping[c] for c in conds])
            y_no_intercept = np.array([mapping[c] for c in conds])

            if remove_random_intercept:
                b_i = result.random_effects.get(rat, None).item()
                y = y - b_i
                y_no_intercept = y_no_intercept - b_i

            ax.plot(
                x,
                y if kw_y_scale == "log2" else (2 ** y),
                color=rat_color[rat],
                alpha=kw_rat_line_alpha,
                linewidth=kw_rat_line_lw,
                zorder=1,
            )
            ax.scatter(
                x,
                y if kw_y_scale == "log2" else (2 ** y),
                color=kw_rat_point_color,
                alpha=kw_rat_point_alpha,
                s=kw_rat_point_size,
                linewidths=0,
                zorder=2,
            )
            Y.append(y)
            Y_no_intercept.append(y_no_intercept)

            # ax.set_xticks(x)
            # ax.set_xticklabels([u.replace("k", "-") for u in conds], rotation=15)

        Y = np.array(Y)
        Y_no_intercept = np.array(Y_no_intercept)
        if show_mean:
            y_mean = np.nanmean(Y, axis=0)
            y_no_intercept_mean = np.nanmean(Y_no_intercept, axis=0)
            ax.plot(
                x,
                y_mean if kw_y_scale == "log2" else (2 ** y_mean),
                # y_no_intercept_mean if kw_y_scale == "log2" else (2 ** y_no_intercept_mean),
                color=kw_mean_color,
                alpha=kw_mean_alpha,
                linewidth=kw_mean_lw,
                marker=kw_mean_marker,
                markersize=np.sqrt(kw_mean_marker_size),
                zorder=10,
                label=LABEL_MEAN
            )

        add_icons(
            ax,
            x_positions=x,
            labels=conds,
            icons_dir=icons_dir,
            icons_params=icons_params,
        )

    ylabel = (
        "Threshold ($\\mu$A, $\\log_2$ scale)"
        if kw_y_scale == "log2"
        else "Threshold ($\\mu$A)"
    )

    _apply_threshold_yaxis(
        axes,
        y_scale=kw_y_scale,
        kw_yticks=kw_yticks,
        kw_ylim=kw_ylim,
        kw_fs=kw_fs,
        kw_ls=kw_ls,
        ylabel_text=ylabel,
    )
    axes[0].legend(
        fontsize=kw_fs,
        frameon=False,
        loc="center",
        bbox_to_anchor=(0.45, 0.95)
    )

    return


def plot_model(
    ax,
    *,
    model,
    result,
    rhs_terms,
    set_reference,
    formula,
    df,
    indicator_terms,
    icons_params,   # (zoom, y-offset)
    reference_term,
    reference_yoffset=-0.21,
    as_percent=True,
    **kws,
):
    icons_dir=ICONS_DIR

    def body_print_ci(name, labels, beta_vec, se_vec, decimals=1):
        beta_vec = np.asarray(beta_vec, dtype=float)
        se_vec = np.asarray(se_vec, dtype=float)

        y_mean = (np.power(2.0, beta_vec) - 1.0) * 100.0
        lo_beta = beta_vec - INFO_SEM * se_vec
        hi_beta = beta_vec + INFO_SEM * se_vec
        lo = (np.power(2.0, lo_beta) - 1.0) * 100.0
        hi = (np.power(2.0, hi_beta) - 1.0) * 100.0

        print()
        print(name)
        print(labels)
        print("mean", np.round(y_mean, decimals))
        print(list(map(tuple, zip(np.round(lo, decimals), np.round(hi, decimals)))))

    ax.clear()
    kw_fs = kws.get("kw_fs")
    kw_ls = kws.get("kw_ls")
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
            raise KeyError()
        beta[k] = params[t]
        se[k] = bse[t]

    if not as_percent:
        raise NotImplementedError

    y = (np.power(2.0, beta) - 1.0) * 100.0

    yerr = None
    k = SEM_TIMES
    beta_m = beta
    se_m = se
    y_m = y
    lo_beta = beta_m - k * se_m
    hi_beta = beta_m + k * se_m
    lo = (np.power(2.0, lo_beta) - 1.0) * 100.0
    hi = (np.power(2.0, hi_beta) - 1.0) * 100.0
    mean_minus_lo = y_m - lo
    hi_minus_mean = hi - y_m
    yerr = np.vstack([mean_minus_lo, hi_minus_mean])

    neg_color = NEG_COLOR
    pos_color = neg_color
    colors = [neg_color if v < 0 else pos_color for v in y]

    is_indicator = np.array([u in indicator_terms for u in terms])
    is_reference = np.array([u == reference_term for u in terms])

    temp_beta = beta[is_indicator & (~is_reference)]
    temp_se = se[is_indicator & (~is_reference)]
    temp_labels = [u for u in indicator_terms if u != reference_term]
    body_print_ci("# shie position", temp_labels, temp_beta, temp_se, decimals=1)

    ax.clear()
    ax.bar(
        x[is_indicator],
        y[is_indicator],
        # color=colors,
        color=FCOLORS["wave"],
        edgecolor="0.2",
        linewidth=0.9,
        zorder=1000,
    )
    ax.errorbar(
        x[is_indicator & (~is_reference)],
        y[is_indicator & (~is_reference)],
        yerr=yerr[:, is_indicator & (~is_reference)],
        fmt="none",
        ecolor="k",
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        zorder=10_000,
    )
    add_icons(
        ax,
        x_positions=x[is_indicator],
        labels=np.array(terms)[is_indicator],
        icons_dir=icons_dir,
        icons_params=icons_params,
        reference_term=reference_term,
        reference_yoffset=reference_yoffset,
        kw_fs=kw_fs,
    )

    beta_is_sh = params[is_sh]
    se_is_sh = bse[is_sh]

    mu_beta = beta_is_sh
    se_beta = se_is_sh

    lo_beta = mu_beta - SEM_TIMES * se_beta
    hi_beta = mu_beta + SEM_TIMES * se_beta

    y_pct  = (np.power(2.0, mu_beta) - 1.0) * 100.0
    lo_pct = (np.power(2.0, lo_beta) - 1.0) * 100.0
    hi_pct = (np.power(2.0, hi_beta) - 1.0) * 100.0
    yerr_sh = np.vstack([y_pct - lo_pct, hi_pct - y_pct])

    body_print_ci("# shie is_hd", ["is_hd"], [mu_beta], [se_beta], decimals=1)

    gap_between = 0.5
    x0 = x[is_indicator][-1] + 1.0 + gap_between
    x_sh = x0 + np.arange(1)
    ax.bar(
        x_sh,
        y_pct,
        # color=colors,
        color=FCOLORS["hd"],
        edgecolor="0.2",
        linewidth=0.9,
        zorder=1000,
    )
    ax.errorbar(
        x_sh,
        y_pct,
        yerr=yerr_sh,
        fmt="none",
        ecolor="k",
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        zorder=10_000
    )

    xticks = x[is_indicator].tolist() + x_sh.tolist()
    id_labels = [""] * len(indicator_terms)
    sh_labels = ["With\nhigh-definition"]
    xticklabels = id_labels + sh_labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        xticklabels,
        rotation=0,
        ha="center",
        fontsize=kw_fs * FS_K,
        linespacing=1.2,
    )
    ax.tick_params(axis="x", which="major", pad=7, rotation=0)

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yticks(list(kw_yticks))
    ax.set_ylabel(
        "% Threshold change vs baseline",
        fontsize=kw_fs,
        linespacing=1.5,
    )
    ax.tick_params(axis="y", left=True, labelleft=True, labelsize=kw_ls)
    ax.set_ylim(-YLO, YLO)

    mid = xticks[1]
    ax.text(
        *(mid, 50),
        LABEL_FIXED,
        # transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=kw_fs
    )

    ax.text(
        *(xticks[3] + .15, 10),
        "Effect of waveforms and polarity",
        # transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=kw_fs,
    )

    ax.text(
        *(xticks[4], -10),
        "Effect of\nhigh-definition",
        # transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=kw_fs,
        linespacing=1.4
    )

    return
