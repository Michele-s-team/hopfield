#!/usr/bin/env python3
"""
Loop-only ratio of the two methods (neighbors vs overlaps).

For each cell (N, rho, alpha, beta) and each method, the median time is fitted
against the number of sweeps S :

        t(S) = t0 + S * t1

t0 is the initialisation cost (everything inside runSweeps_* that does not
scale with S), t1 is the cost of one Metropolis sweep. The plotted ratio is
built from the slopes t1 only, so the initialisation is removed.

One heatmap (alpha vs rho, log scale centred on 1) per beta and per CSV kind.
A table of the fits (t0, t1, init fraction, S needed for init < 1 %) is saved
next to the CSVs as loopfit_<kind>.csv.

Several CSV files (one per n_sweeps) or one CSV with several n_sweeps are fine:
they are concatenated. Missing files / columns / betas are skipped, never a crash.

Usage:
  python make_plots_loop.py --dir results/
  python make_plots_loop.py --bits a.csv b.csv c.csv d.csv
  python make_plots_loop.py --dir results/ --ratio nb_over_ov --min-points 3

Plot range: edit ALPHA_RANGE and RHO_RANGE at the top of the file.
"""

import argparse
import glob
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

# Plot range (inclusive). None = no restriction.
#   e.g. ALPHA_RANGE = (0.05, 0.3)    RHO_RANGE = (0.5, 1.0)
ALPHA_RANGE = (0.05, 0.35)
RHO_RANGE = (0.5, 0.8)
KEY = ["N", "rho", "alpha", "beta"]
PRETTY = {"rho": r"$\rho$", "alpha": r"$\alpha$", "beta": r"$\beta$", "N": r"$N$"}

# kind -> (glob pattern for the default file names, column prefix, label tag)
KINDS = {
    "bits":          ("speedup_bits*.csv",          "t_bitwise",       "bits"),
    "nobits_indep":  ("speedup_nobits_indep*.csv",  "t_classic_indep", "classic,indep"),
    "nobits_shared": ("speedup_nobits_shared*.csv", "t_classic_shared", "classic,shared"),
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


def restrict(df, alpha_range, rho_range):
    """Keep only rows with alpha / rho inside [min, max] (inclusive).

    Returns the filtered frame and a file-name suffix describing the range
    ("" if no range is given), so restricted plots never overwrite full ones.
    """
    suffix = ""
    for col, rng in (("rho", rho_range), ("alpha", alpha_range)):
        if rng is None:
            continue
        lo, hi = sorted(rng)
        df = df[(df[col] >= lo - 1e-9) & (df[col] <= hi + 1e-9)]
        suffix += f"_{col}{lo:g}-{hi:g}"
    return df, suffix


def make_pivot(sub, x, y, zcol):
    piv = sub.pivot_table(index=y, columns=x, values=zcol, aggfunc="mean")
    return piv.sort_index().sort_index(axis=1)


def fit_cells(df, c_nb, c_ov, min_points):
    """Weighted linear fit t = t0 + S*t1 per cell and per method.

    The timing noise is relative (about constant in %), so the weights are
    1/t, which prevents the longest runs from dominating the fit.
    """
    d = df[KEY + ["n_sweeps", c_nb, c_ov]].copy()
    for k in ("rho", "alpha", "beta"):
        d[k] = d[k].astype(float).round(9)
    # same cell and same n_sweeps in several files -> median
    d = d.groupby(KEY + ["n_sweeps"], as_index=False)[[c_nb, c_ov]].median()

    rows = []
    for key, g in d.groupby(KEY):
        row = dict(zip(KEY, key))
        row["n_points"] = int(g["n_sweeps"].nunique())
        S = g["n_sweeps"].to_numpy(float)
        for tag, col in (("nb", c_nb), ("ov", c_ov)):
            t0 = t1 = frac = s1 = np.nan
            t = g[col].to_numpy(float)
            if row["n_points"] >= min_points and np.all(np.isfinite(t)) and np.all(t > 0):
                t1, t0 = np.polyfit(S, t, 1, w=1.0 / t)
                frac = t0 / (t0 + t1 * S.max()) if t1 > 0 else np.nan
                # n_sweeps for which t0 / t(S) < 1 %
                s1 = max(99.0 * t0 / t1, 0.0) if t1 > 0 else np.nan
            row[f"t0_{tag}"] = t0
            row[f"t1_{tag}"] = t1
            row[f"init_frac_Smax_{tag}"] = frac
            row[f"S_for_1pct_{tag}"] = s1
        row["S_min"], row["S_max"] = S.min(), S.max()
        rows.append(row)
    return pd.DataFrame(rows)


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


# --------------------------------------------------------------------------- plot

def heatmap_ratio(df, zcol, name, label, title, outdir, curve=True, center=1.0):
    x, y = "rho", "alpha"
    piv = make_pivot(df, x, y, zcol)
    Z = piv.values
    vals = Z[np.isfinite(Z)]

    if vals.size == 0 or piv.shape[0] < 2 or piv.shape[1] < 2:
        print(f"  [skip] {name}: not enough points for a 2D grid")
        return False
    if vals.min() <= 0:
        print(f"  [skip] {name}: non-positive values, log scale impossible")
        return False

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

    above = np.where(np.isfinite(Z), Z >= center, np.nan)
    if cell_boundary(ax, above, color="black", lw=2.0, zorder=5):
        handles.append(Line2D([], [], color="black", lw=2.0,
                              label=rf"$Z={center:g}$"))

    if curve:
        side = np.where((5/6)*Y[:, None] <= (X - 0.5)[None, :] + 1e-9, 1.0, 0.0)
        if side.min() != side.max() and cell_boundary(
                ax, side, color="crimson", lw=2.0, zorder=6, ls="--"):
            handles.append(Line2D([], [], color="crimson", lw=2.0,
                                  label=r"$\rho=(2/3)*\alpha+1/2$"))

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
    return True


def ratio_label(tag, ratio):
    num, den = ("overlaps", "neighbors") if ratio == "ov_over_nb" else ("neighbors", "overlaps")
    return (r"$t^{\rm " + tag + r"}_{\rm " + num + r"}\,/\,t^{\rm " + tag
            + r"}_{\rm " + den + r"}$" + "\n(Metropolis loop only)")


# --------------------------------------------------------------------------- per-kind driver

def find_files(kind, given, a):
    if given:
        return list(given)
    return sorted(glob.glob(os.path.join(a.dir, KINDS[kind][0])))


def process_kind(kind, given, a):
    pattern, prefix, tag = KINDS[kind]
    files = find_files(kind, given, a)
    print(f"\n[{kind}] {len(files)} file(s)")
    if not files:
        print("  [skip] no file found")
        return False

    c_nb = f"{prefix}_neighbors_{a.stat}"
    c_ov = f"{prefix}_overlaps_{a.stat}"
    needed = KEY + ["n_sweeps", c_nb, c_ov]

    frames = []
    for p in files:
        if not os.path.isfile(p):
            print(f"  [skip] {p}: not found")
            continue
        try:
            d = pd.read_csv(p)
        except Exception as e:
            print(f"  [skip] {p}: cannot read ({e})")
            continue
        miss = [c for c in needed if c not in d.columns]
        if miss:
            print(f"  [skip] {p}: missing columns {', '.join(miss)}")
            continue
        print(f"  read {p}: {len(d)} rows, n_sweeps = "
              f"{', '.join(str(s) for s in sorted(d['n_sweeps'].unique()))}")
        frames.append(d[needed])
    if not frames:
        print("  [skip] no usable file")
        return False

    df = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    n_s = sorted(df["n_sweeps"].unique())
    print(f"  n_sweeps available: {', '.join(str(int(s)) for s in n_s)}")
    if len(n_s) < max(a.min_points, 2):
        print(f"  [skip] need at least {max(a.min_points, 2)} distinct n_sweeps")
        return False

    fits = fit_cells(df, c_nb, c_ov, a.min_points)
    outdir = os.path.dirname(os.path.abspath(files[0]))
    fit_path = os.path.join(outdir, f"loopfit_{kind}.csv")
    try:
        fits.to_csv(fit_path, index=False, float_format="%.6g")
        print(f"  -> {os.path.basename(fit_path)} (t0, t1, init fraction per cell)")
    except Exception as e:
        print(f"  [warn] cannot write {fit_path}: {e}")

    if a.ratio == "ov_over_nb":
        fits["ratio_loop"] = fits["t1_ov"] / fits["t1_nb"]
    else:
        fits["ratio_loop"] = fits["t1_nb"] / fits["t1_ov"]
    fits = fits.replace([np.inf, -np.inf], np.nan)

    N_sel = select(fits, {"N": a.N_fixed})
    if N_sel.empty:
        print(f"  [skip] no rows for N={a.N_fixed:g}")
        return False

    N_sel, suffix = restrict(N_sel, ALPHA_RANGE, RHO_RANGE)
    if N_sel.empty:
        print("  [skip] no rows left after the alpha / rho restriction")
        return False
    if suffix:
        print(f"  range restriction: {suffix.strip('_').replace('_', ', ')}")

    betas = np.sort(N_sel["beta"].unique())
    if a.betas:
        betas = [b for b in betas if any(np.isclose(b, bb) for bb in a.betas)]
        if not betas:
            print("  [skip] none of the requested betas is in this file")
            return False
    print(f"  betas: {', '.join(f'{b:g}' for b in betas)}")

    label = ratio_label(tag, a.ratio)
    Nt = rf"$N={a.N_fixed:g}$"
    ok = False

    for b in betas:
        sub = select(N_sel, {"beta": b})

        # Summary of the initialisation cost for this beta.
        msg = []
        for m in ("nb", "ov"):
            fr = sub[f"init_frac_Smax_{m}"].dropna()
            s1 = sub[f"S_for_1pct_{m}"].dropna()
            if len(fr):
                msg.append(f"{m}: init/t(Smax) median {100*fr.median():.2f}% "
                           f"max {100*fr.max():.2f}%, S for <1% up to {s1.max():.0f}")
        if msg:
            print(f"  beta={b:g} | " + " | ".join(msg))

        name = f"ratio_loop_{kind}_alpha_rho_beta{b:g}"
        try:
            ok |= bool(heatmap_ratio(sub, "ratio_loop", name, label,
                                     rf"{Nt}, $\beta={b:g}$", outdir,
                                     curve=not a.no_curve))
        except Exception as e:
            print(f"  [error] {name}: {e}", file=sys.stderr)
            if a.debug:
                traceback.print_exc()
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".", help="Directory searched for speedup_<kind>*.csv")
    ap.add_argument("--bits", nargs="*", default=None, help="CSV file(s) speedup_bits")
    ap.add_argument("--indep", nargs="*", default=None, help="CSV file(s) speedup_nobits_indep")
    ap.add_argument("--shared", nargs="*", default=None, help="CSV file(s) speedup_nobits_shared")
    ap.add_argument("--stat", default="median", choices=["median", "mean"])
    ap.add_argument("--N-fixed", type=float, default=N_DEFAULT)
    ap.add_argument("--betas", nargs="*", type=float,
                    help="Betas to plot (default: all betas found)")
    ap.add_argument("--min-points", type=int, default=3,
                    help="Minimum number of distinct n_sweeps per cell for the fit (default 3)")
    ap.add_argument("--ratio", default="ov_over_nb", choices=["ov_over_nb", "nb_over_ov"],
                    help="ov_over_nb: t_overlaps/t_neighbors (default); "
                         "nb_over_ov: t_neighbors/t_overlaps (as in the C++ column)")
    ap.add_argument("--no-curve", action="store_true", help="Do not draw rho = alpha + 1/2")
    ap.add_argument("--no-tex", action="store_true")
    ap.add_argument("--debug", action="store_true", help="Print tracebacks")
    a = ap.parse_args()

    set_style(not a.no_tex)

    given = {"bits": a.bits, "nobits_indep": a.indep, "nobits_shared": a.shared}
    done = 0
    for kind in KINDS:
        try:
            done += bool(process_kind(kind, given[kind], a))
        except Exception as e:  # one kind never kills the others
            print(f"  [error] {kind}: {e}", file=sys.stderr)
            if a.debug:
                traceback.print_exc()

    if done == 0:
        print("\nNo figure produced (no usable CSV found).", file=sys.stderr)
    else:
        print(f"\nDone: {done}/3 kinds processed.")


if __name__ == "__main__":
    main()
