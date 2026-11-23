#!/usr/bin/env python3
"""
Plot 1 only: ratio t_overlaps / t_neighbors, alpha (y) vs rho (x),
for 3 CSV files (bits, nobits_indep, nobits_shared), one figure per beta.
A missing CSV / column / beta is skipped with a message, never a crash.

Usage:
  python make_plots.py
  python make_plots.py --dir results/
  python make_plots.py --bits a.csv --indep b.csv --shared c.csv
  python make_plots.py --N-fixed 1024 --stat mean --betas 0.5 1 2
"""

import argparse
import os
import shutil
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import FuncNorm

N_DEFAULT = 32 * 32
PRETTY = {"rho": r"$\rho$", "alpha": r"$\alpha$", "beta": r"$\beta$", "N": r"$N$"}

# kind -> (default csv name, column prefix, colorbar label)
KINDS = {
    "bits": (
        "speedup_bits.csv", "t_bitwise",
        r"$t^{\rm bits}_{\rm overlaps}\,/\,t^{\rm bits}_{\rm neighbors}$",
    ),
    "nobits_indep": (
        "speedup_nobits_indep.csv", "t_classic_indep",
        r"$t^{\rm classic,\,indep}_{\rm overlaps}\,/\,t^{\rm classic,\,indep}_{\rm neighbors}$",
    ),
    "nobits_shared": (
        "speedup_nobits_shared.csv", "t_classic_shared",
        r"$t^{\rm classic,\,shared}_{\rm overlaps}\,/\,t^{\rm classic,\,shared}_{\rm neighbors}$",
    ),
}


# --------------------------------------------------------------------------- style

def latex_ok():
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
        print("  (usetex disabled -> mathtext used)", file=sys.stderr)
    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.formatter.use_mathtext": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 1.2,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def pretty(name):
    return PRETTY.get(name, name.replace("_", " "))


def fmt_val(v, strip=False):
    s = f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}"
    return s.rstrip("0").rstrip(".") if strip and "." in s else s


def save(fig, outdir, name):
    kw = dict(bbox_inches="tight", pad_inches=0.05)
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


# --------------------------------------------------------------------------- color scale

def log_centered_norm(vmin, vmax, c=1.0):
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
        lower, upper = lx < 0, lx > 0
        if lmin < 0:
            result[lower] = 0.5 - 0.5 * lx[lower] / lmin
        if lmax > 0:
            result[upper] = 0.5 + 0.5 * lx[upper] / lmax
        return result

    def inv(y):
        y = np.asarray(y, dtype=float)
        result = np.full_like(y, c)
        lower, upper = y < 0.5, y > 0.5
        if lmin < 0:
            result[lower] = c * np.exp((0.5 - y[lower]) * 2 * lmin)
        if lmax > 0:
            result[upper] = c * np.exp((y[upper] - 0.5) * 2 * lmax)
        return result

    return FuncNorm((fwd, inv), vmin=vmin, vmax=vmax)


def centered_ticks(vmin, vmax, c, n=4):
    lo = np.geomspace(vmin, c, n)[:-1] if vmin < c else np.array([])
    hi = np.geomspace(c, vmax, n) if vmax > c else np.array([c])
    return np.unique(np.concatenate([lo, hi]))


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


# --------------------------------------------------------------------------- plot 1

def heatmap_ratio(df, zcol, name, label, title, outdir, agg, center=1.0):
    x, y = "rho", "alpha"
    piv = make_pivot(df, x, y, zcol, agg)
    Z = piv.values
    vals = Z[np.isfinite(Z)]

    if vals.size == 0 or piv.shape[0] < 2 or piv.shape[1] < 2:
        print(f"  [skip] {name}: not enough points for a 2D grid")
        return
    if vals.min() <= 0:
        print(f"  [skip] {name}: non-positive values, log scale impossible")
        return

    vmin = min(float(vals.min()), center)
    vmax = max(float(vals.max()), center)
    if vmin == vmax:
        vmin, vmax = center / 1.001, center * 1.001

    norm = log_centered_norm(vmin, vmax, center)
    cmap = plt.get_cmap("RdBu_r")
    ticks = centered_ticks(vmin, vmax, center)

    ny, nx = Z.shape
    X, Y = piv.columns.values.astype(float), piv.index.values.astype(float)

    fig, ax = plt.subplots(figsize=(
        np.clip(0.35 * nx + 3.0, 4.6, 8),
        np.clip(0.25 * ny + 2.2, 3.4, 6),
    ))

    mesh = ax.pcolormesh(
        np.arange(nx), np.arange(ny), np.ma.masked_invalid(Z),
        shading="nearest", cmap=cmap, norm=norm,
        edgecolors="face", linewidth=0.5, rasterized=False,
    )

    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xticks(np.arange(nx))
    ax.set_xticklabels([f"{v:g}" for v in X], rotation=45 if nx > 8 else 0)
    ax.set_yticks(np.arange(ny))
    ax.set_yticklabels([f"{v:g}" for v in Y])
    ax.tick_params(axis="both", which="both", length=0)
    ax.minorticks_off()

    handles = []

    # Boundary Z >= 1 / Z < 1
    above = np.where(np.isfinite(Z), Z >= center, np.nan)
    if cell_boundary(ax, above, color="black", lw=2.0, zorder=5):
        handles.append(Line2D([], [], color="black", lw=2.0,
                              label=rf"$Z={center:g}$"))

    # Curve rho = alpha + 1/2
    side = np.where(Y[:, None] <= (X - 0.5)[None, :] + 1e-9, 1.0, 0.0)
    if side.min() != side.max() and cell_boundary(
            ax, side, color="crimson", lw=2.0, zorder=6, ls="--"):
        handles.append(Line2D([], [], color="crimson", lw=2.0,
                              label=r"$\rho=\alpha+1/2$"))

    if handles:
        ax.legend(handles=handles, loc="upper right",
                  framealpha=0.9, edgecolor="0.7")

    ax.set_xlabel(pretty(x))
    ax.set_ylabel(pretty(y))
    ax.set_title(title, fontsize=10)

    cb = fig.colorbar(mesh, ax=ax, shrink=0.92, pad=0.03, aspect=25)
    cb.outline.set_linewidth(0.8)
    cb.set_ticks(ticks)
    cb.set_ticklabels([fmt_val(t, strip=True) for t in ticks])
    cb.minorticks_off()
    cb.ax.tick_params(direction="out", length=4, width=0.8, right=True, labelsize=9)
    cb.set_label(label)

    fig.tight_layout()
    save(fig, outdir, name)


# --------------------------------------------------------------------------- per-CSV driver

def process_csv(kind, path, a):
    default_name, prefix, label = KINDS[kind]
    print(f"\n[{kind}] {path}")

    if not os.path.isfile(path):
        print("  [skip] file not found")
        return False

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  [skip] cannot read file: {e}")
        return False

    c_ov = f"{prefix}_overlaps_{a.stat}"
    c_ne = f"{prefix}_neighbors_{a.stat}"
    needed = ["rho", "alpha", "beta", "N", c_ov, c_ne]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"  [skip] missing columns: {', '.join(missing)}")
        return False

    zcol = f"ratio_{kind}"
    df[zcol] = df[c_ov] / df[c_ne]
    df = df.replace([np.inf, -np.inf], np.nan)

    N_sel = select(df, {"N": a.N_fixed})
    if N_sel.empty:
        print(f"  [skip] no rows for N={a.N_fixed:g}")
        return False

    betas = np.sort(N_sel["beta"].unique())
    if a.betas:
        betas = [b for b in betas if any(np.isclose(b, bb) for bb in a.betas)]
        if not betas:
            print("  [skip] none of the requested betas is in this file")
            return False
    print(f"  betas: {', '.join(f'{b:g}' for b in betas)}")

    outdir = os.path.dirname(os.path.abspath(path))
    Nt = rf"$N={a.N_fixed:g}$"
    ok = False

    for b in betas:
        sub = select(N_sel, {"beta": b})
        name = f"ratio_{kind}_alpha_rho_beta{b:g}"
        try:
            heatmap_ratio(sub, zcol, name, label,
                          rf"{Nt}, $\beta={b:g}$", outdir, a.agg)
            ok = True
        except Exception as e:
            print(f"  [error] {name}: {e}", file=sys.stderr)
            if a.debug:
                traceback.print_exc()
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".", help="Directory for default CSV names")
    ap.add_argument("--bits", default=None, help="CSV speedup_bits")
    ap.add_argument("--indep", default=None, help="CSV speedup_nobits_indep")
    ap.add_argument("--shared", default=None, help="CSV speedup_nobits_shared")
    ap.add_argument("--agg", default="mean", choices=["mean", "median"])
    ap.add_argument("--stat", default="median", choices=["median", "mean"])
    ap.add_argument("--N-fixed", type=float, default=N_DEFAULT)
    ap.add_argument("--betas", nargs="*", type=float,
                    help="Betas to plot (default: all betas found in each CSV)")
    ap.add_argument("--no-tex", action="store_true")
    ap.add_argument("--debug", action="store_true", help="Print tracebacks")
    a = ap.parse_args()

    set_style(not a.no_tex)

    paths = {
        "bits": a.bits or os.path.join(a.dir, KINDS["bits"][0]),
        "nobits_indep": a.indep or os.path.join(a.dir, KINDS["nobits_indep"][0]),
        "nobits_shared": a.shared or os.path.join(a.dir, KINDS["nobits_shared"][0]),
    }

    done = 0
    for kind, path in paths.items():
        try:
            done += bool(process_csv(kind, path, a))
        except Exception as e:  # last safety net: one CSV never kills the others
            print(f"  [error] {kind}: {e}", file=sys.stderr)
            if a.debug:
                traceback.print_exc()

    if done == 0:
        print("\nNo figure produced (no usable CSV found).", file=sys.stderr)
    else:
        print(f"\nDone: {done}/3 CSV processed.")


if __name__ == "__main__":
    main()