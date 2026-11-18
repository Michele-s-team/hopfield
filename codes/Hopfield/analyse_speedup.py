#!/usr/bin/env python3
"""
Publication-quality color maps of a quantity Z as a function of a pair of
parameters (X, Y), from the classic vs bitwise benchmark CSV.

Key behaviours
  * The color range is adapted to the data actually displayed (after --fix / aggregation).
  * Speedups / ratios (auto-detected, or --center 1) use a diverging colormap
    centered on 1, on a log scale: red = slower than baseline (<1), blue = faster (>1),
    with a bold line along the cell edges separating >1 from <1.

Examples:
  python plot_colormap.py bench.csv --list

  # speedup vs rho and alpha, N=1024  (auto: diverging map centered on 1)
  python plot_colormap.py bench.csv -z speedup_overlaps_indep -x rho -y alpha --fix N=1024

  # ratio of two speedups computed on the fly
  python plot_colormap.py bench.csv -z "speedup_overlaps_indep/speedup_neighbors_indep" \
      -x rho -y alpha --fix N=1024 beta=5

  # one panel per N, shared color scale, vector output for a paper
  python plot_colormap.py bench.csv -z speedup_neighbors_shared -x rho -y alpha --facet N --out fig.pdf

  # a time (not a ratio): sequential colormap, log scale
  python plot_colormap.py bench.csv -z t_bitwise_overlaps_mean -x N -y rho --center none --log
"""
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import FuncNorm, LogNorm, Normalize, TwoSlopeNorm

PARAMS = ["N", "rho", "alpha", "P", "beta", "k_eff", "n_sweeps"]
PRETTY = {"rho": r"$\rho$", "alpha": r"$\alpha$", "beta": r"$\beta$",
          "k_eff": r"$k_{\mathrm{eff}}$", "n_sweeps": "sweeps", "N": r"$N$", "P": r"$P$"}


# --------------------------------------------------------------------------- style
def set_style():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 11, "axes.labelsize": 13, "axes.titlesize": 12,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.linewidth": 0.8,
    })


def pretty(name):
    return PRETTY.get(name, name.replace("_", " "))


def fmt_val(v, strip=False):
    """Compact number format: 2 decimals <10, 1 decimal <100, none above."""
    s = f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}"
    return s.rstrip("0").rstrip(".") if strip and "." in s else s


# --------------------------------------------------------------------------- data
def parse_fix(items):
    fixed = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"--fix expects key=value, got: {it!r}")
        k, v = it.split("=", 1)
        fixed[k.strip()] = float(v)
    return fixed


def select(df, fixed):
    mask = np.ones(len(df), dtype=bool)
    for k, v in fixed.items():
        if k not in df.columns:
            sys.exit(f"Unknown column in --fix: {k}")
        mask &= np.isclose(df[k].astype(float), v)
    return df[mask]


def make_pivot(sub, x, y, agg):
    piv = sub.pivot_table(index=y, columns=x, values="_z", aggfunc=agg)
    return piv.sort_index().sort_index(axis=1)


# --------------------------------------------------------------------------- color scale
def log_centered_norm(vmin, vmax, c=1.0):
    """Piecewise-log norm: `c` is exactly the middle of the colormap; each side uses
    the full half-colormap, so the range adapts to the data on both sides."""
    lmin, lmax = np.log(vmin / c), np.log(vmax / c)

    def fwd(x):
        lx = np.log(np.clip(np.asarray(x, dtype=float), 1e-300, None) / c)
        return np.where(lx >= 0, 0.5 + 0.5 * lx / lmax, 0.5 - 0.5 * lx / lmin)

    def inv(y):
        y = np.asarray(y, dtype=float)
        ly = np.where(y >= 0.5, (y - 0.5) * 2 * lmax, (0.5 - y) * 2 * lmin)
        return c * np.exp(ly)

    return FuncNorm((fwd, inv), vmin=vmin, vmax=vmax)


def centered_ticks(vmin, vmax, c, log, n=4):
    sp = np.geomspace if log else np.linspace
    lo = sp(vmin, c, n)[:-1] if vmin < c else np.array([])
    hi = sp(c, vmax, n) if vmax > c else np.array([c])
    return np.unique(np.concatenate([lo, hi]))


# --------------------------------------------------------------------------- drawing
def draw(ax, piv, norm, cmap, center, annotate):
    X, Y, Z = piv.columns.values, piv.index.values, piv.values
    nx, ny = len(X), len(Y)
    Zm = np.ma.masked_invalid(Z)
    mesh = ax.pcolormesh(np.arange(nx), np.arange(ny), Zm, shading="nearest",
                         cmap=cmap, norm=norm, edgecolors="white", linewidth=1.0)
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xticks(np.arange(nx))
    ax.set_xticklabels([f"{v:g}" for v in X], rotation=45 if nx > 8 else 0)
    ax.set_yticks(np.arange(ny))
    ax.set_yticklabels([f"{v:g}" for v in Y])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # bold boundary along cell edges between cells above and below the center value
    if center is not None:
        above = np.where(np.isfinite(Z), Z >= center, np.nan)
        kw = dict(color="black", lw=2.6, solid_capstyle="round", zorder=5)
        for i in range(ny):
            for j in range(nx - 1):
                if np.isfinite(above[i, j]) and np.isfinite(above[i, j + 1]) \
                        and above[i, j] != above[i, j + 1]:
                    ax.plot([j + 0.5] * 2, [i - 0.5, i + 0.5], **kw)
        for i in range(ny - 1):
            for j in range(nx):
                if np.isfinite(above[i, j]) and np.isfinite(above[i + 1, j]) \
                        and above[i, j] != above[i + 1, j]:
                    ax.plot([j - 0.5, j + 0.5], [i + 0.5] * 2, **kw)

    if annotate:
        for i in range(ny):
            for j in range(nx):
                v = Z[i, j]
                if np.isfinite(v):
                    r, g, b, _ = cmap(norm(v))
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    ax.text(j, i, fmt_val(v), ha="center", va="center", fontsize=8.5,
                            color="white" if lum < 0.45 else "black")
    return mesh


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("-z", help="column or pandas expression (e.g. a/b)")
    ap.add_argument("-x", help="parameter on the x axis")
    ap.add_argument("-y", help="parameter on the y axis")
    ap.add_argument("--fix", nargs="*", default=[], metavar="K=V",
                    help="fixed parameters, e.g. N=1024 beta=5")
    ap.add_argument("--facet", help="one panel per value of this parameter")
    ap.add_argument("--agg", default="mean", choices=["mean", "median", "min", "max"])
    ap.add_argument("--center", default="auto",
                    help="'auto' (1 if Z looks like a speedup/ratio), 'none', or a number")
    ap.add_argument("--cmap", default=None,
                    help="default: RdBu if centered (red<c<blue), viridis otherwise")
    ap.add_argument("--linear", action="store_true",
                    help="linear instead of log scale when centered")
    ap.add_argument("--asym", action="store_true",
                    help="adapt each side of the center separately (default: symmetric, "
                         "so x and 1/x get equally intense colors)")
    ap.add_argument("--log", action="store_true", help="log scale (non-centered maps)")
    ap.add_argument("--vmin", type=float)
    ap.add_argument("--vmax", type=float)
    ap.add_argument("--annotate", action=argparse.BooleanOptionalAction, default=None,
                    help="values in cells (default: auto, on for small grids)")
    ap.add_argument("--label", help="custom colorbar/title label for Z")
    ap.add_argument("--out", help="output file (.png/.pdf/.svg); otherwise show window")
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--list", action="store_true", help="list columns and parameter values")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)

    if a.list or not (a.z and a.x and a.y):
        print("Columns:")
        for c in df.columns:
            print("  ", c)
        print("\nParameter values:")
        for p in PARAMS + ["n_rep"]:
            if p in df.columns:
                print(f"  {p}: {sorted(df[p].unique())}")
        if not a.list:
            print("\n-z, -x and -y are required to plot.")
        return

    for c in (a.x, a.y) + ((a.facet,) if a.facet else ()):
        if c not in df.columns:
            sys.exit(f"Unknown column: {c}")

    fixed = parse_fix(a.fix)
    df = select(df, fixed)
    if df.empty:
        sys.exit("No rows left after --fix filtering.")
    try:
        df["_z"] = df.eval(a.z)
    except Exception as e:
        sys.exit(f"Cannot evaluate -z {a.z!r}: {e}")

    used = {a.x, a.y, a.facet, *fixed}
    leftover = [p for p in PARAMS if p in df.columns and p not in used and p != "k_eff"
                and df[p].nunique() > 1]
    if leftover:
        print(f"Warning: these parameters still vary and are aggregated ({a.agg}): "
              f"{', '.join(leftover)}  -> use --fix or --facet", file=sys.stderr)

    # ---- pivots first, so the color range adapts to what is actually displayed
    groups = [(None, df)] if not a.facet else list(df.groupby(a.facet))
    pivots = [(v, make_pivot(g, a.x, a.y, a.agg)) for v, g in groups]
    allvals = np.concatenate([p.values.ravel() for _, p in pivots])
    allvals = allvals[np.isfinite(allvals)]
    if allvals.size == 0:
        sys.exit("No finite values to plot.")
    vmin = a.vmin if a.vmin is not None else allvals.min()
    vmax = a.vmax if a.vmax is not None else allvals.max()

    # ---- center
    c = a.center.lower()
    if c == "auto":
        center = 1.0 if any(k in a.z.lower() for k in ("speedup", "ratio")) else None
    elif c == "none":
        center = None
    else:
        center = float(c)

    # ---- norm + colormap
    if center is not None:
        log = not a.linear and center > 0 and vmin > 0
        vmin, vmax = min(vmin, center * 0.999), max(vmax, center * 1.001)
        if not a.asym and log:
            L = max(vmax / center, center / vmin)
            vmin, vmax = center / L, center * L
        elif not a.asym:
            d = max(vmax - center, center - vmin)
            vmin, vmax = center - d, center + d
        norm = (log_centered_norm(vmin, vmax, center) if log
                else TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax))
        cmap = plt.get_cmap(a.cmap or "RdBu")
    else:
        log = a.log
        if log and vmin <= 0:
            sys.exit("Log scale impossible: values <= 0.")
        norm = LogNorm(vmin, vmax) if log else Normalize(vmin, vmax)
        cmap = plt.get_cmap(a.cmap or "viridis")

    # ---- figure layout
    set_style()
    n = len(pivots)
    ncols, nrows = min(n, 3), int(np.ceil(n / min(n, 3)))
    nx = max(len(p.columns) for _, p in pivots)
    ny = max(len(p.index) for _, p in pivots)
    pw, ph = np.clip(0.75 * nx + 1.8, 4.5, 9), np.clip(0.6 * ny + 1.5, 3.8, 8)
    fig, axes = plt.subplots(nrows, ncols, figsize=(pw * ncols + 1.2, ph * nrows + 0.8),
                             squeeze=False, constrained_layout=True)

    annotate = a.annotate if a.annotate is not None else (nx * ny <= 120)
    mesh = None
    for ax, (val, piv) in zip(axes.ravel(), pivots):
        if len(piv.columns) < 2 or len(piv.index) < 2:
            ax.text(0.5, 0.5, "not enough points\nfor a 2D grid", ha="center",
                    va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        mesh = draw(ax, piv, norm, cmap, center, annotate)
        ax.set_xlabel(pretty(a.x))
        ax.set_ylabel(pretty(a.y))
        if val is not None:
            ax.set_title(f"{pretty(a.facet)} = {val:g}", pad=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    label = a.label or a.z.replace("_", " ")
    if mesh is not None:
        cb = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02, aspect=28)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=3)
        if center is not None:
            ticks = centered_ticks(vmin, vmax, center, log)
            cb.set_ticks(ticks)
            cb.set_ticklabels([fmt_val(t, strip=True) for t in ticks])
            cb.minorticks_off()
        cb.set_label(label, labelpad=8)

    sub = "   ".join(f"{pretty(k)} = {v:g}" for k, v in fixed.items())
    fig.suptitle(label + (f"\n{sub}" if sub else ""), fontsize=13, fontweight="bold")

    if a.out:
        fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
        print(f"Figure saved: {a.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()