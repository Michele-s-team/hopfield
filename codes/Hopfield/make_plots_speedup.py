#!/usr/bin/env python3
"""
Figures benchmark classic vs bitwise -- all in one file.
Style: same as analyse_speedup.py (Computer Modern, inward ticks,
plasma for N, viridis for alpha, PDF + PNG at 300 dpi).
Figures are saved in the same directory as the CSV.

Usage:
  python make_plots.py bench.csv
  python make_plots.py bench.csv --alpha 0.1 --N-fixed 1024 --logy
"""

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from matplotlib.colors import FuncNorm, Normalize

N_DEFAULT = 32 * 32
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]
PRETTY = {
    "rho": r"$\rho$",
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "N": r"$N$",
    "P": r"$P$",
    "n_sweeps": r"$N_{\rm sweep}$",
}
EXTRA_PARAMS = ["P", "n_sweeps"]
BOTTOM = 0.15


# --------------------------------------------------------------------------- style

def latex_ok():
    """Check whether LaTeX is installed and can compile a test expression."""
    if shutil.which("latex") is None:
        return False
    try:
        with plt.rc_context({"text.usetex": True}):
            fig = plt.figure(figsize=(1, 1))
            fig.text(0.5, 0.5, r"$\alpha$ lp")
            fig.canvas.draw()
            plt.close(fig)
        return True
    except Exception:
        return False


def set_style(use_tex=True):
    use_tex = use_tex and latex_ok()
    if not use_tex:
        print("  (usetex disabled -> mathtext 'cm' used)", file=sys.stderr)

    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.formatter.use_mathtext": True,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,

        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.handlelength": 2.0,
        "legend.labelspacing": 0.3,
        "legend.handletextpad": 0.5,

        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.2,

        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def pretty(name):
    return PRETTY.get(name, name.replace("_", " "))


def fmt_val(v, strip=False):
    s = f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}"
    return s.rstrip("0").rstrip(".") if strip and "." in s else s


def save(fig, outdir, name, tight=False):
    kw = dict(bbox_inches="tight", pad_inches=0.05) if tight else dict(pad_inches=0.05)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), **kw)
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300, **kw)
    print(f"  -> {name}.pdf / .png")
    plt.close(fig)


# --------------------------------------------------------------------------- data

def select(df, fixed):
    mask = np.ones(len(df), dtype=bool)
    for k, v in fixed.items():
        mask &= np.isclose(df[k].astype(float), v)
    return df[mask]


def make_pivot(sub, x, y, zcol, agg):
    piv = sub.pivot_table(index=y, columns=x, values=zcol, aggfunc=agg)
    return piv.sort_index().sort_index(axis=1)


# --------------------------------------------------------------------------- color scales

def log_centered_norm(vmin, vmax, c=1.0):
    """Piecewise logarithmic normalization with c at the colormap midpoint."""
    if vmin <= 0 or vmax <= 0 or c <= 0:
        raise ValueError("Logarithmic normalization requires positive values.")

    if not (vmin <= c <= vmax):
        raise ValueError("The center must lie between vmin and vmax.")

    lmin = np.log(vmin / c)
    lmax = np.log(vmax / c)

    def fwd(x):
        x = np.asarray(x, dtype=float)
        lx = np.log(np.clip(x, 1e-300, None) / c)
        result = np.full_like(lx, 0.5)

        lower = lx < 0
        upper = lx > 0

        if lmin < 0:
            result[lower] = 0.5 - 0.5 * lx[lower] / lmin

        if lmax > 0:
            result[upper] = 0.5 + 0.5 * lx[upper] / lmax

        return result

    def inv(y):
        y = np.asarray(y, dtype=float)
        result = np.full_like(y, c)

        lower = y < 0.5
        upper = y > 0.5

        if lmin < 0:
            result[lower] = c * np.exp((0.5 - y[lower]) * 2 * lmin)

        if lmax > 0:
            result[upper] = c * np.exp((y[upper] - 0.5) * 2 * lmax)

        return result

    return FuncNorm((fwd, inv), vmin=vmin, vmax=vmax)


def centered_ticks(vmin, vmax, c, n=4):
    """Generate logarithmically spaced colorbar ticks around c."""
    lo = np.geomspace(vmin, c, n)[:-1] if vmin < c else np.array([])
    hi = np.geomspace(c, vmax, n) if vmax > c else np.array([c])
    return np.unique(np.concatenate([lo, hi]))


# --------------------------------------------------------------------------- colormap

def cell_boundary(ax, cls, **kw):
    ny, nx = cls.shape
    kw.setdefault("solid_capstyle", "butt")
    drawn = False

    for i in range(ny):
        for j in range(nx - 1):
            if np.isfinite(cls[i, j]) and np.isfinite(cls[i, j + 1]) \
                    and cls[i, j] != cls[i, j + 1]:
                ax.plot([j + 0.5] * 2, [i - 0.5, i + 0.5], **kw)
                drawn = True

    for i in range(ny - 1):
        for j in range(nx):
            if np.isfinite(cls[i, j]) and np.isfinite(cls[i + 1, j]) \
                    and cls[i, j] != cls[i + 1, j]:
                ax.plot([j - 0.5, j + 0.5], [i + 0.5] * 2, **kw)
                drawn = True

    return drawn


def heatmap(
    df, x, y, zcol, name, label, title, outdir, agg,
    center=1.0, curve=None, diverging=False
):
    """
    Plot a heatmap.

    Sequential maps use a linear normalization over the actual data range.
    The diverging map uses a logarithmic normalization centered on center.
    """
    piv = make_pivot(df, x, y, zcol, agg)
    Z = piv.values
    vals = Z[np.isfinite(Z)]

    if vals.size == 0 or piv.shape[0] < 2 or piv.shape[1] < 2:
        print(f"  [skip] {name}: not enough points for a 2D grid")
        return

    data_min = float(vals.min())
    data_max = float(vals.max())

    if diverging:
        if data_min <= 0:
            raise ValueError(
                f"{name}: logarithmic normalization requires positive values."
            )

        vmin = min(data_min, center)
        vmax = max(data_max, center)

        if vmin == vmax:
            vmin = center / 1.001
            vmax = center * 1.001

        norm = log_centered_norm(vmin, vmax, center)
        cmap = plt.get_cmap("RdBu_r")
        ticks = centered_ticks(vmin, vmax, center)

    else:
        vmin = data_min
        vmax = data_max

        if vmin == vmax:
            delta = max(abs(vmin) * 1e-9, 1e-12)
            vmin -= delta
            vmax += delta

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("viridis")
        ticks = np.linspace(vmin, vmax, 5)

    ny, nx = Z.shape
    X, Y = piv.columns.values.astype(float), piv.index.values.astype(float)

    fig, ax = plt.subplots(
        figsize=(
            np.clip(0.35 * nx + 3.0, 4.6, 8),
            np.clip(0.25 * ny + 2.2, 3.4, 6),
        )
    )

    mesh = ax.pcolormesh(
        np.arange(nx),
        np.arange(ny),
        np.ma.masked_invalid(Z),
        shading="nearest",
        cmap=cmap,
        norm=norm,
        edgecolors="face",
        linewidth=0.5,
        rasterized=False,
    )

    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)

    ax.set_xticks(np.arange(nx))
    ax.set_xticklabels(
        [f"{v:g}" for v in X],
        rotation=45 if nx > 8 else 0,
    )

    ax.set_yticks(np.arange(ny))
    ax.set_yticklabels([f"{v:g}" for v in Y])

    # Keep numerical tick labels but remove the tick marks.
    ax.tick_params(axis="both", which="both", length=0)
    ax.minorticks_off()

    # Draw the boundary between values above and below the reference value
    # only on the diverging map.
    handles = []

    # Boundary between Z >= center and Z < center (diverging map only).
    if diverging:
        above = np.where(np.isfinite(Z), Z >= center, np.nan)
        if cell_boundary(ax, above, color="black", lw=2.0, zorder=5):
            handles.append(
                Line2D([], [], color="black", lw=2.0,
                       label=rf"$Z={center:g}$")
            )

    # Optional curve (e.g. rho = alpha + 1/2), along cell boundaries.
    if curve is not None:
        f, clabel = curve
        side = np.where(Y[:, None] <= f(X)[None, :] + 1e-9, 1.0, 0.0)

        if side.min() == side.max():
            print(f"  [info] {name}: the curve does not separate the grid")
        elif cell_boundary(ax, side, color="crimson", lw=2.0, zorder=6, ls="--"):
            handles.append(
                Line2D([], [], color="crimson", lw=2.0, label=clabel)
            )

    if handles:
        ax.legend(
            handles=handles,
            loc="upper right",
            framealpha=0.9,
            edgecolor="0.7",
        )

    ax.set_xlabel(pretty(x))
    ax.set_ylabel(pretty(y))
    ax.set_title(title, fontsize=10)

    cb = fig.colorbar(mesh, ax=ax, shrink=0.92, pad=0.03, aspect=25)
    cb.outline.set_linewidth(0.8)
    cb.set_ticks(ticks)
    cb.set_ticklabels([fmt_val(t, strip=True) for t in ticks])
    cb.minorticks_off()

    # Colorbar tick marks point outward.
    cb.ax.tick_params(
        direction="out",
        length=4,
        width=0.8,
        right=True,
        labelsize=9,
    )

    cb.set_label(label)

    fig.tight_layout()
    save(fig, outdir, name, tight=True)


def best_method_map(df, x, y, name, title, outdir, agg, stat):
    """
    Categorical map showing the fastest bitwise method in each cell.

    Method times are first averaged over the remaining parameters
    (beta, etc.), then compared.
    """
    d = df.assign(
        _ov=df[f"t_bitwise_overlaps_{stat}"],
        _ne=df[f"t_bitwise_neighbors_{stat}"],
    )

    t_ov_bits = make_pivot(d, x, y, "_ov", agg)
    t_ne_bits = make_pivot(d, x, y, "_ne", agg)

    if t_ov_bits.shape[0] < 2 or t_ov_bits.shape[1] < 2:
        print(f"  [skip] {name}: not enough points for a 2D grid")
        return

    Z = np.where(
        np.isfinite(t_ov_bits.values) & np.isfinite(t_ne_bits.values),
        (t_ov_bits.values <= t_ne_bits.values).astype(float),
        np.nan,
    )

    ny, nx = Z.shape
    X, Y = t_ov_bits.columns.values, t_ov_bits.index.values
    c_ov, c_ne = "C0", "#ff7f0e"

    fig, ax = plt.subplots(
        figsize=(
            np.clip(0.35 * nx + 3.0, 4.6, 8),
            np.clip(0.25 * ny + 2.2, 3.4, 6),
        )
    )

    ax.pcolormesh(
        np.arange(nx),
        np.arange(ny),
        np.ma.masked_invalid(Z),
        shading="nearest",
        cmap=ListedColormap([c_ne, c_ov]),
        vmin=0,
        vmax=1,
        edgecolors="face",
        linewidth=0.5,
        rasterized=False,
    )

    cell_boundary(ax, Z, color="black", lw=2.0, zorder=5)

    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)

    ax.set_xticks(np.arange(nx))
    ax.set_xticklabels(
        [f"{v:g}" for v in X],
        rotation=45 if nx > 8 else 0,
    )

    ax.set_yticks(np.arange(ny))
    ax.set_yticklabels([f"{v:g}" for v in Y])

    # Keep numerical tick labels but remove the tick marks.
    ax.tick_params(axis="both", which="both", length=0)
    ax.minorticks_off()

    ax.set_xlabel(pretty(x))
    ax.set_ylabel(pretty(y))
    ax.set_title(title, fontsize=10)

    ax.legend(
        handles=[
            Patch(
                facecolor=c_ov,
                edgecolor="k",
                linewidth=0.8,
                label="overlaps",
            ),
            Patch(
                facecolor=c_ne,
                edgecolor="k",
                linewidth=0.8,
                label="neighbors",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        framealpha=0.9,
        edgecolor="0.7",
    )

    fig.tight_layout()
    save(fig, outdir, name, tight=True)


# --------------------------------------------------------------------------- curves

def curves(
    df, groupcol, zcol, name, ylabel, title, outdir, agg, cmap_name, logy,
    xcol="beta", ncol=1
):
    groups = sorted(df[groupcol].unique())

    if df[xcol].nunique() < 2 or not groups:
        print(f"  [skip] {name}: not enough points")
        return

    palette = plt.get_cmap(cmap_name)(np.linspace(0.1, 0.85, len(groups)))
    fig, ax = plt.subplots(figsize=(4, 3.2))

    for i, g in enumerate(groups):
        s = (
            df[np.isclose(df[groupcol], g)]
            .groupby(xcol)[zcol]
            .agg(agg)
            .sort_index()
        )

        fmt = f"{g:g}" if groupcol == "N" else f"{g:.2f}"

        ax.plot(
            s.index,
            s.values,
            marker=MARKERS[i % len(MARKERS)],
            markersize=4,
            color=palette[i],
            linewidth=1.2,
            label=rf"${groupcol if groupcol == 'N' else chr(92) + groupcol}={fmt}$",
        )

    lo, hi = ax.get_ylim()

    if lo < 1 < hi:
        ax.axhline(1.0, color="k", lw=0.6, ls=":", zorder=0)

    if logy:
        ax.set_yscale("log")
    else:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_xlabel(pretty(xcol))
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="major", alpha=0.25, linewidth=0.5)

    ax.legend(
        framealpha=0.9,
        edgecolor="0.7",
        handlelength=1.8,
        ncol=ncol,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=BOTTOM)
    save(fig, outdir, name)


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("csv")
    ap.add_argument("--agg", default="mean", choices=["mean", "median"])
    ap.add_argument(
        "--N-fixed",
        type=float,
        default=N_DEFAULT,
        help=f"Fixed N (default {N_DEFAULT})",
    )
    ap.add_argument("--rho", type=float, default=1.0, help="Fixed rho (default 1)")
    ap.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Fixed alpha for curves with several N values (default: median alpha)",
    )
    ap.add_argument("--Ns", nargs="*", type=float, help="N values to plot (default: all)")
    ap.add_argument(
        "--alphas",
        nargs="*",
        type=float,
        help="Alpha values to plot (default: all)",
    )
    ap.add_argument("--logy", action="store_true", help="Use a logarithmic y-axis for curves")
    ap.add_argument(
        "--stat",
        default="median",
        choices=["median", "mean"],
        help="Bitwise time statistic used for the ratio and best-method selection "
             "(default: median, as for the CSV speedups)",
    )
    ap.add_argument("--no-tex", action="store_true", help="Disable usetex")

    a = ap.parse_args()

    set_style(not a.no_tex)

    df = pd.read_csv(a.csv)
    outdir = os.path.dirname(os.path.abspath(a.csv))

    # Derived quantities.
    t_ov_bits = df[f"t_bitwise_overlaps_{a.stat}"]
    t_ne_bits = df[f"t_bitwise_neighbors_{a.stat}"]

    t_ov_nobits = df[f"t_classic_indep_overlaps_{a.stat}"]
    t_ne_nobits = df[f"t_classic_indep_neighbors_{a.stat}"]

    df["ratio_tbits"] = t_ov_bits / t_ne_bits
    df["ratio_tnobits"] = t_ov_nobits / t_ne_nobits

    ov_best_bits = t_ov_bits <= t_ne_bits

    for kind in ("indep", "shared"):
        df[f"speedup_best_{kind}"] = np.where(
            ov_best_bits,
            df[f"speedup_overlaps_{kind}"],
            df[f"speedup_neighbors_{kind}"],
        )

    # Selections.
    rho_sel = select(df, {"rho": a.rho})
    N_sel = select(df, {"N": a.N_fixed})
    both = select(rho_sel, {"N": a.N_fixed})

    if a.alpha is None:
        av = np.sort(df["alpha"].unique())
        a.alpha = float(av[len(av) // 2])

    if rho_sel.empty or N_sel.empty:
        sys.exit("No rows for this rho / N: check --rho and --N-fixed.")

    varying = [
        p for p in EXTRA_PARAMS
        if p in df.columns and df[p].nunique() > 1
    ]

    if varying:
        print(
            f"Warning: {', '.join(varying)} vary and are being averaged.",
            file=sys.stderr,
        )

    Nt = rf"$N={a.N_fixed:g}$"
    rt = rf"$\rho={a.rho:g}$"

    avg = (
        rf"(average over $\beta$)"
        if a.agg == "mean"
        else r"(median over $\beta$)"
    )

    ratio_lab_bits = (
        r"$t^{\rm bits}_{\rm overlaps}\,/\,"
        r"t^{\rm bits}_{\rm neighbors}$"
    )

    ratio_lab_nobits = (
            r"$t^{\rm classic}_{\rm overlaps}\,/\,"
            r"t^{\rm classic}_{\rm neighbors}$"
        )

    print("Figures:")

    # 1-2. Ratio of bitwise overlaps / neighbors times.
    # Only this heatmap uses a logarithmic diverging scale centered on 1.
    heatmap(
        N_sel,
        "rho",
        "alpha",
        "ratio_tnobits",
        "ratio_tnobits_indep_alpha_rho",
        ratio_lab_nobits,
        f"{Nt} {avg}",
        outdir,
        a.agg,
        curve=(lambda r: r - 0.5, r"$\rho=\alpha+1/2$"),
        diverging=True,
    )

    heatmap(
        N_sel,
        "rho",
        "alpha",
        "ratio_tbits",
        "ratio_tbits_alpha_rho",
        ratio_lab_bits,
        f"{Nt} {avg}",
        outdir,
        a.agg,
        curve=(lambda r: r - 0.5, r"$\rho=\alpha+1/2$"),
        diverging=True,
    )

    # This heatmap uses a sequential linear scale.
    heatmap(
        rho_sel,
        "N",
        "alpha",
        "ratio_tbits",
        "ratio_tbits_alpha_N",
        ratio_lab_bits,
        f"{rt} {avg}",
        outdir,
        a.agg,
    )

    # 2bis. Fastest bitwise method, alpha (y) vs rho (x), fixed N.
    # Uncomment to generate the categorical map.
    #
    # best_method_map(
    #     N_sel,
    #     "rho",
    #     "alpha",
    #     "best_method_alpha_rho",
    #     rf"Fastest bitwise method, {Nt} {avg}",
    #     outdir,
    #     a.agg,
    #     a.stat,
    # )

    # 3-4. Speedup of the best method, alpha (y) vs N (x).
    # These heatmaps use sequential linear scales.
    for kind in ("indep", "shared"):
        heatmap(
            rho_sel,
            "N",
            "alpha",
            f"speedup_best_{kind}",
            f"speedup_best_{kind}_alpha_N",
            f"Speed-up factor ({kind}, best method)",
            f"{rt} {avg}",
            outdir,
            a.agg,
        )

    # 5-6. Curves vs beta, several N values (fixed alpha and rho).
    sub = select(rho_sel, {"alpha": a.alpha})

    if a.Ns:
        sub = sub[np.isin(sub["N"], a.Ns)]

    for kind in ("indep", "shared"):
        curves(
            sub,
            "N",
            f"speedup_best_{kind}",
            f"speedup_best_{kind}_vs_beta_N",
            f"Speed-up factor ({kind})",
            rf"{rt}, $\alpha={a.alpha:g}$",
            outdir,
            a.agg,
            "plasma",
            a.logy,
            ncol=2,
        )

    # 7-8. Curves vs beta, several alpha values (fixed N and rho).
    sub = both

    if a.alphas:
        sub = sub[np.isin(sub["alpha"], a.alphas)]

    for kind in ("indep", "shared"):
        curves(
            sub,
            "alpha",
            f"speedup_best_{kind}",
            f"speedup_best_{kind}_vs_beta_alpha",
            f"Speed-up factor ({kind})",
            rf"{rt}, {Nt}",
            outdir,
            a.agg,
            "viridis",
            a.logy,
        )


if __name__ == "__main__":
    main()