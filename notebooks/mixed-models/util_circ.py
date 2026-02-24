# util_circ.py
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

is_gr = "is_ground"
is_bip = "is_bipolar"

dm = "distance"
bip_dm = "bip_distance"
inv_dm = "inverse_distance"

th = "th"
th_rad = "th_rad"
cos = "cos2"
sin = "sin2"

SEM_TIMES = 1
NEG_COLOR = "#4C78A8"
FS_K = 0.7

ICONS_DIR = "/home/vishu/bits"

YLO = 65
YPAD = .1
LABEL_MEAN = "Average threshold across 8 rats\n$\downarrow$ Lower is more effective"
LABEL_FIXED = "Fixed effects\n$\downarrow$ Lower is more effective"

_FS_K = 0.75

FCOLORS = dict(
    pos="#E69F00",
    dist="#7A5195",
    size="#2CA02C",
    orient="#1F77B4",
    hd="#8C564B",
    wave="#7F7F7F",
)


def add_theta_icons(
    ax,
    x_positions,
    theta_rads,
    *,
    icons_params=(0.12, -0.12),  # (size, y_offset) in *axes* fraction
    spine_alpha=0.35,
    spine_lw=0.9,
    label_alpha=0.85,
):
    size, yoff = icons_params
    ax.figure.canvas.draw()

    for xi, th in zip(x_positions, theta_rads):
        x_axes = ax.transAxes.inverted().transform(
            ax.transData.transform((float(xi), 0.0))
        )[0]
        x0 = x_axes - 0.5 * size
        y0 = yoff

        axp = ax.inset_axes((x0, y0, size, size), projection="polar")
        axp.set_theta_zero_location("E")
        axp.set_theta_direction(1)
        axp.set_xticks([])
        axp.set_yticks([])
        axp.grid(False)

        # show circular spine
        for spine in axp.spines.values():
            spine.set_alpha(spine_alpha)
            spine.set_linewidth(spine_lw)
            spine.set_edgecolor("0.25")

        t = float(th) % (2.0 * np.pi)
        axp.annotate(
            "",
            xy=(t, 1.0),
            xytext=(t, 0.0),
            arrowprops=dict(arrowstyle="-|>", lw=2.0, color="0.15"),
        )

        deg = (np.rad2deg(t)) % 360.0
        axp.text(
            0.5, -0.22,
            f"{deg:.0f}°",
            transform=axp.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="0.25",
            alpha=label_alpha,
        )


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
                fontsize=kw_fs * _FS_K,
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


def add_angle_icons(ax, x_positions, angles_deg, icons_dir, icons_params):
    ax.set_xticks(x_positions)
    zoom, yoff = icons_params
    for xi, ang in zip(x_positions, angles_deg):
        src = os.path.join(icons_dir, f"{int(ang)}.png")
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
    kw_mean_lw = kws.get("kw_mean_lw", 2.6)
    kw_mean_marker = kws.get("kw_mean_marker", "o")
    kw_mean_marker_size = kws.get("kw_mean_marker_size", 18)

    diameters = ["SE-NW", "S-N", "NE-SW", "E-W"]
    radii = ["SE-C", "S-C", "SW-C", "W-C", "NW-C", "N-C", "NE-C", "E-C"]
    vertices = ["-C", "-SE", "-S", "-SW", "-W", "-NW", "-N", "-NE", "-E"]
    # diameters = ["E-W", "SE-NW", "S-N", "NE-SW"]
    # radii = ["W-C", "NW-C", "N-C", "NE-C", "E-C", "SE-C", "S-C", "SW-C"]
    # vertices = ["-W", "-NW", "-N", "-NE", "-E",  "-SE", "-S", "-SW", "-C"]
    groups = [diameters, radii, vertices]

    rats = sorted(df.rat.unique())
    if color_by_rat:
        cmap = plt.get_cmap(kw_rat_cmap)
        ts = np.linspace(0.20, 0.85, len(rats))
        rat_color = {r: cmap(t) for r, t in zip(rats, ts)}
    else:
        rat_color = {r: "0.70" for r in rats}

    for j, (ax, conds) in enumerate(zip(axes, groups)):
        ax.clear()
        x = np.arange(len(conds), dtype=float)
        Y = []

        for rat in rats:
            sub = df.loc[df.rat == rat].reset_index(drop=True)
            assert sub.shape[0] == 21
            mapping = dict(zip(sub.label, sub.a))

            y = np.array([mapping[c] for c in conds], dtype=float)

            if remove_random_intercept:
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
    axes[1].legend(
        fontsize=kw_fs,
        frameon=False,
        loc="center",
        bbox_to_anchor=(0.5, 0.95)
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
    orientation_style="bars",
    **kws,
):
    icons_dir=ICONS_DIR

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

    xticks = x[is_indicator].tolist() + x_distance.tolist()
    id_labels = [""] * len(indicator_terms)
    distance_labels = [f"Electrode\ndistance\n{u} (mm)" for u in distance]
    xticklabels = id_labels + distance_labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        xticklabels,
        rotation=0,
        ha="center",
        fontsize=kw_fs * _FS_K,
        linespacing=1.2,
    )
    ax.tick_params(axis="x", which="major", pad=7, rotation=0)

    if orientation_style == "bars":
        fe = result.fe_params
        cov = result.cov_params().loc[fe.index, fe.index].values
        idx = {k: i for i, k in enumerate(fe.index)}

        a0 = np.zeros(len(fe));   a0[idx[cos]] = 1.0
        a45 = np.zeros(len(fe));  a45[idx[sin]] = 1.0
        a90 = -a0
        a135 = -a45
        A = np.vstack([a0, a45, a90, a135])

        mu = A @ fe.values
        se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", A, cov, A), 0.0))
        lo = mu - SEM_TIMES * se
        hi = mu + SEM_TIMES * se

        y_or  = (2.0**mu - 1.0) * 100.0
        lo_or = (2.0**lo - 1.0) * 100.0
        hi_or = (2.0**hi - 1.0) * 100.0
        yerr_or = np.vstack([y_or - lo_or, hi_or - y_or])

        x1 = x_distance[-1] + 1.0 + gap_between
        x_or = x1 + np.arange(4)

        ax.bar(
            x_or,
            y_or,
            # color=colors[:4],
            color=FCOLORS["orient"],
            edgecolor="0.2",
            linewidth=0.9,
            zorder=1000
        )
        ax.errorbar(
            x_or, y_or, yerr=yerr_or, fmt="none",
            ecolor="k",
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
            zorder=10_000
        )

        add_angle_icons(
            ax,
            x_positions=x_or,
            angles_deg=[0, 45, 90, 135],
            icons_dir=icons_dir,
            icons_params=icons_params,
        )

        xticks = x[is_indicator].tolist() + x_distance.tolist() + x_or.tolist()
        xticklabels = id_labels + distance_labels + [""] * len(x_or)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, ha="center", fontsize=kw_fs * _FS_K, linespacing=1.2)
        ax.tick_params(axis="x", which="major", pad=7, rotation=0)
    
    elif "bars-polar" in orientation_style:
        fe = result.fe_params
        cov = result.cov_params().loc[fe.index, fe.index].values
        idx = {k: i for i, k in enumerate(fe.index)}

        bc = params[cos]
        bs = params[sin]

        theta0 = 0.5 * np.arctan(bs / bc)
        if bc < 0.0:
            theta0 = theta0 + (np.pi / 2)
        else:
            raise NotImplementedError

        t0 = theta0
        t1 = (t0 + 0.5 * np.pi) % (2.0 * np.pi)
        print(np.rad2deg(t0))
        print(np.rad2deg(t1))

        w0 = np.array([np.cos(2.0 * t0), np.sin(2.0 * t0)])
        w1 = np.array([np.cos(2.0 * t1), np.sin(2.0 * t1)])

        mu = np.array([
            w0[0] * bc + w0[1] * bs,
            w1[0] * bc + w1[1] * bs,
        ])

        i_cos = idx[cos]
        i_sin = idx[sin]
        cov2 = cov[np.ix_([i_cos, i_sin], [i_cos, i_sin])]

        se = np.array([
            np.sqrt(w0 @ cov2 @ w0),
            np.sqrt(w1 @ cov2 @ w1),
        ])

        lo = mu - SEM_TIMES * se
        hi = mu + SEM_TIMES * se

        y_or  = (2.0**mu - 1.0) * 100.0
        lo_or = (2.0**lo - 1.0) * 100.0
        hi_or = (2.0**hi - 1.0) * 100.0
        yerr_or = np.vstack([y_or - lo_or, hi_or - y_or])

        if "both" in orientation_style:
            x1 = x_distance[-1] + 1.0 + gap_between
            x_or = x1 + np.arange(2)
            ax.bar(
                x_or,
                y_or,
                # color=colors[:2],
                color=FCOLORS["orient"],
                edgecolor="0.2",
                linewidth=0.9,
                zorder=1000
            )
            ax.errorbar(
                x_or, y_or, yerr=yerr_or, fmt="none",
                ecolor="k", elinewidth=1.5, capsize=4, capthick=1.5, zorder=10_000
            )
            add_angle_icons(
                ax,
                x_positions=x_or,
                angles_deg=[97, 7],
                icons_dir=icons_dir,
                icons_params=icons_params,
            )

        else:
            x1 = x_distance[-1] + 1.0 + gap_between
            x_or = x1 + np.arange(1)
            ax.bar(
                x_or,
                y_or[1:],
                # color=colors[1:],
                color=FCOLORS["orient"],
                edgecolor="0.2",
                linewidth=0.9,
                zorder=1000
            )
            ax.errorbar(
                x_or, y_or[1:], yerr=yerr_or[..., 1:], fmt="none",
                ecolor="k", elinewidth=1.5, capsize=4, capthick=1.5, zorder=10_000
            )
            add_angle_icons(
                ax,
                x_positions=x_or,
                angles_deg=[7],
                icons_dir=icons_dir,
                icons_params=icons_params,
            )

        xticks = x[is_indicator].tolist() + x_distance.tolist() + x_or.tolist()
        xticklabels = id_labels + distance_labels + [""] * len(x_or)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, ha="center", fontsize=kw_fs * _FS_K, linespacing=1.2)
        ax.tick_params(axis="x", which="major", pad=7, rotation=0)

    else:
        raise ValueError(
            f"orientation_style must be 'bars' or 'bars-polar', got {orientation_style!r}"
        )

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yticks(list(kw_yticks))
    ax.set_ylabel(
        "% Threshold change vs baseline",
        fontsize=kw_fs,
        linespacing=1.5,
    )
    ax.tick_params(axis="y", left=True, labelleft=True, labelsize=kw_ls)
    ax.set_ylim(-YLO, YLO)

    mid = xticks[2]
    ax.text(
        *(mid, 45),
        LABEL_FIXED,
        # transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=kw_fs
    )

    ax.text(
        *(8, 10),
        "Effect of position",
        # transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=kw_fs,
        linespacing=1.4
    )

    lo, hi = xticks[-3:-1]
    mid = (lo + hi) / 2
    ax.text(
        *(mid, -10),
        "Effect of distance",
        # transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=kw_fs,
    )

    mid = xticks[-1]
    ax.text(
        *(mid, 10),
        "Effect of\norientation",
        # transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=kw_fs,
    )

    return


def draw_orientation_polar(
    ax_parent,
    *,
    theta0_rad: float,
    pct_plus: float,
    pct_minus: float,
    inset_bounds=(0.86, 0.08, 0.12, 0.34),  # (x0, y0, w, h) in axes fraction
    kw_fs=10,
):
    # Polar inset
    axp = ax_parent.inset_axes(inset_bounds, projection="polar")
    axp.set_theta_zero_location("E")   # 0° at East
    axp.set_theta_direction(1)         # CCW positive
    axp.set_xticks([])
    axp.set_yticks([])
    axp.grid(True, alpha=0.25)
    for spine in axp.spines.values():
        spine.set_alpha(0.25)
    
    # Wrap angles
    t0 = float(theta0_rad) % (2.0 * np.pi)
    t1 = (t0 + 0.5 * np.pi) % (2.0 * np.pi)

    # Normalize lengths just for display (keep sign in labels)
    L0 = abs(float(pct_plus))
    L1 = abs(float(pct_minus))
    Lmax = max(L0, L1, 1e-12)
    r0 = 1.0 * (L0 / Lmax)
    r1 = 1.0 * (L1 / Lmax)

    # Draw arrows (vectors)
    # Use annotate so it looks like an arrow, not a line
    axp.annotate(
        "",
        xy=(t0, r0),
        xytext=(t0, 0.0),
        arrowprops=dict(arrowstyle="-|>", lw=2.0, color="0.15"),
    )
    axp.annotate(
        "",
        xy=(t1, r1),
        xytext=(t1, 0.0),
        arrowprops=dict(arrowstyle="-|>", lw=2.0, color="0.50"),
    )

    # Labels near arrow tips
    lab0 = f"+{pct_plus:.1f}%"
    lab1 = f"{pct_minus:.1f}%"
    axp.text(t0, min(1.15, r0 + 0.12), lab0, ha="center", va="center", fontsize=kw_fs * 0.75)
    axp.text(t1, min(1.15, r1 + 0.12), lab1, ha="center", va="center", fontsize=kw_fs * 0.75)

    # Small title
    axp.text(
        0.5, 1.15, r"Orientation",
        transform=axp.transAxes,
        ha="center", va="bottom",
        fontsize=kw_fs * 0.8,
        color="0.25",
    )
