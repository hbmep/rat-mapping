import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from util_circ import (
    is_gr, is_bip,
    dm, bip_dm, inv_dm,
    SEM_TIMES, NEG_COLOR, FS_K, ICONS_DIR,
    _apply_threshold_yaxis,
    YLO, YPAD, LABEL_MEAN, LABEL_FIXED, FCOLORS
)

is_large = "is_large"

# YLO = 55
FS_K = .88


def turn_off(ax):
    sides = ['left', 'bottom', 'top', 'right']
    ax.spines[sides].set_visible(False)
    ax.tick_params(
        axis="both", left=False, labelleft=False,
        labelbottom=False, bottom=False
    )


def combo_to_icon_name(is_large, label):
    s = str(label)
    s = s[1:] if s.startswith("-") else s
    s = s.replace("-", "_")
    return f"{int(is_large)}_{s}"


def add_combo_icons(
    ax,
    x_positions,
    combos,
    icons_dir,
    icons_params,
    jitter_sign=1.0,
):
    zoom, yoff, jitter = icons_params
    ax.set_xticks(x_positions)
    ax.set_xticklabels([""] * len(combos))
    jit = [((-1) ** i) * jitter_sign for i in range(len(combos))]
    jit = [u * jitter for u in jit]
    for zi, (xi, (is_large, lab)) in enumerate(zip(x_positions, combos)):
        fname = combo_to_icon_name(is_large, lab)
        src = os.path.join(icons_dir, f"{fname}.png")
        if not os.path.exists(src):
            print(f"[icon] missing: {src}")
            continue
        img = mpl.image.imread(src)
        imagebox = mpl.offsetbox.OffsetImage(img, zoom=zoom)
        ab = mpl.offsetbox.AnnotationBbox(
            imagebox,
            (xi, yoff + jit[zi]),
            xycoords=ax.get_xaxis_transform(),
            frameon=False,
            box_alignment=(0.5, 1.0),
            pad=0.0,
        )
        ax.add_artist(ab)


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
    icons_dir=ICONS_DIR

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
    kw_mean_lw = kws.get("kw_mean_lw", 1.6)
    kw_mean_marker = kws.get("kw_mean_marker", "o")
    kw_mean_marker_size = kws.get("kw_mean_marker_size", 18)

    group1 = [
        (0, "-LL"),
        (0, "-L"),
        (0, "-LM"),
        (0, "-LM2"),
        (0, "-M"),
    ]   # small to ground (5)
    group2 = [
        (0, "L-LL"),
        (0, "LM-L"),
        (0, "LM2-LM"),
        (0, "M-LM2"),   # small consecutive (4)
        (0, "LM-LL"),
        (0, "LM2-L"),
        (0, "M-LM"),    # small with one skip (3)
        (0, "LM2-LL"),
        (0, "M-L"),     # small with two skips (2)
        (0, "M-LL"),    # small with three skips (1)
    ]
    group3 = [
        (1, "-LL"),
        (1, "-LM"),
        (1, "-M"),      # large to ground (3)
        (1, "LM-LL"),
        (1, "M-LM"),    # large consecutive (2)
        (1, "M-LL"),    # large one skip (1)
    ]
    groups = [group1, group2, group3]
    rats = sorted(df["rat"].unique())

    if color_by_rat:
        cmap = plt.get_cmap(kw_rat_cmap)
        ts = np.linspace(0.20, 0.85, len(rats))
        rat_color = {r: cmap(t) for r, t in zip(rats, ts)}
    else:
        rat_color = {r: "0.70" for r in rats}

    for j, (ax, combos) in enumerate(zip(axes, groups)):
        ax.clear()
        x = np.arange(len(combos), dtype=float)
        Y = []

        for rat in rats:
            sub = df.loc[df.rat == rat].reset_index(drop=True)
            assert sub.shape[0] <= 21
            mapping = {(int(r.is_large), str(r.label)): float(r.a) for r in sub.itertuples(index=False)}

            y = np.full(len(combos), np.nan, dtype=float)
            for ci, (is_large, lab) in enumerate(combos):
                y[ci] = mapping.get((int(is_large), str(lab)), np.nan)

            if remove_random_intercept:
                if result is None:
                    raise ValueError("remove_random_intercept=True requires `result` (MixedLM fit).")
                b_i = result.random_effects.get(rat, None).item()
                y = y - b_i

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

        Y = np.array(Y)
        if show_mean:
            y_mean = np.nanmean(Y, axis=0)
            ax.plot(
                x,
                y_mean if kw_y_scale == "log2" else (2 ** y_mean),
                color=kw_mean_color,
                alpha=kw_mean_alpha,
                linewidth=kw_mean_lw,
                marker=kw_mean_marker,
                markersize=np.sqrt(kw_mean_marker_size),
                zorder=10,
                label=LABEL_MEAN if j == 1 else None
            )

        ip = icons_params[j]
        add_combo_icons(
            ax,
            x_positions=x,
            combos=combos,
            icons_dir=icons_dir,
            icons_params=ip[:-1],       # (zoom, yoff, jitter)
            jitter_sign=ip[-1],         # sign
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
    axes[1].legend(
        fontsize=kw_fs,
        frameon=False,
        loc="center",
        bbox_to_anchor=(0.5, 0.95)
    )

    return


def add_indicator_icons_size(
    ax,
    *,
    x_positions,
    indicator_terms,
    icons_dir,
    icons_params,
    reference_term=None,
    reference_yoffset=-0.21,
    reference_text="Baseline",
    reference_text_kwargs=None,
):
    zoom, yoff = icons_params

    if reference_text_kwargs is None:
        reference_text_kwargs = dict(ha="center", va="top")

    for xi, term in zip(x_positions, indicator_terms):
        fname = term
        src = os.path.join(icons_dir, f"0_{fname}.png")
        if not os.path.exists(src):
            print(f"[icon] missing: {src}")
            continue

        img = mpl.image.imread(src)
        imagebox = mpl.offsetbox.OffsetImage(img, zoom=zoom)
        ab = mpl.offsetbox.AnnotationBbox(
            imagebox,
            (xi, yoff),
            xycoords=ax.get_xaxis_transform(),
            frameon=False,
            box_alignment=(0.5, 1.0),
            pad=0.0,
        )
        ax.add_artist(ab)

        if reference_term is not None and term == reference_term:
            ax.text(
                xi,
                reference_yoffset,
                reference_text,
                transform=ax.get_xaxis_transform(),
                **reference_text_kwargs,
            )


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

    ax.clear()
    kw_fs = kws.get("kw_fs", 10)
    kw_ls = kws.get("kw_ls", 10)
    kw_yticks = kws.get("kw_yticks_model", (-60, -40, -20, 0, 20, 40, 60))

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

    ax.clear()
    ax.bar(
        x[is_indicator],
        y[is_indicator],
        # color=colors,
        color=FCOLORS["pos"],
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
    add_indicator_icons_size(
        ax,
        x_positions=x[is_indicator],
        indicator_terms=indicator_terms,
        icons_dir=icons_dir,
        icons_params=icons_params,
        reference_term=reference_term,
        reference_yoffset=reference_yoffset,
        reference_text="(Baseline)",
        reference_text_kwargs=dict(fontsize=kw_fs * FS_K, ha="center", va="top"),
    )

    beta_inv_dm = params[inv_dm]
    se_inv_dm = bse[inv_dm]

    distance = sorted(df[dm].unique())
    distance = np.array([u for u in distance if not np.isnan(u)])
    inverse_distance = 1. / distance

    mu_beta = beta_inv_dm * inverse_distance
    se_beta = se_inv_dm * inverse_distance

    lo_beta = mu_beta - SEM_TIMES * se_beta
    hi_beta = mu_beta + SEM_TIMES * se_beta

    y_pct  = (np.power(2.0, mu_beta) - 1.0) * 100.0
    lo_pct = (np.power(2.0, lo_beta) - 1.0) * 100.0
    hi_pct = (np.power(2.0, hi_beta) - 1.0) * 100.0
    yerr_distance = np.vstack([y_pct - lo_pct, hi_pct - y_pct])

    gap_between = 0.5
    x0 = x[is_indicator][-1] + 1.0 + gap_between
    x_distance = x0 + np.arange(len(distance))
    ax.bar(
        x_distance,
        y_pct,
        # color=colors,
        color=FCOLORS["dist"],
        edgecolor="0.2",
        linewidth=0.9,
        zorder=1000,
    )
    ax.errorbar(
        x_distance,
        y_pct,
        yerr=yerr_distance,
        fmt="none",
        ecolor="k",
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        zorder=10_000
    )

    beta_is_large = params[is_large]
    se_is_large = bse[is_large]

    mu_beta = beta_is_large
    se_beta = se_is_large

    lo_beta = mu_beta - SEM_TIMES * se_beta
    hi_beta = mu_beta + SEM_TIMES * se_beta

    y_pct  = (np.power(2.0, mu_beta) - 1.0) * 100.0
    lo_pct = (np.power(2.0, lo_beta) - 1.0) * 100.0
    hi_pct = (np.power(2.0, hi_beta) - 1.0) * 100.0
    yerr_sh = np.vstack([y_pct - lo_pct, hi_pct - y_pct])

    x1 = x_distance[-1] + 1.0 + gap_between
    x_large = x1 + np.arange(1)
    ax.bar(
        x_large,
        y_pct,
        # color=colors,
        color=FCOLORS["size"],
        edgecolor="0.2",
        linewidth=0.9,
        zorder=1000,
    )
    ax.errorbar(
        x_large,
        y_pct,
        yerr=yerr_sh,
        fmt="none",
        ecolor="k",
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        zorder=10_000
    )

    xticks = x[is_indicator].tolist() + x_distance.tolist() + x_large.tolist()
    id_labels = [""] * len(indicator_terms)
    distance_labels = [f"Electrode\ndistance\n{u} (mm)" for u in distance]
    size_labels = ["Larger electode size"]
    xticklabels = id_labels + distance_labels + size_labels
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
        *(xticks[4] + .15, 10),
        "Effect of position",
        # transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=kw_fs,
    )

    ax.text(
        *(xticks[8] + .15, -10),
        "Effect of distance",
        # transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=kw_fs,
    )

    mid = xticks[-1]
    ax.text(
        *(mid, 10),
        "Effect of size",
        # transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=kw_fs,
    )

    return
