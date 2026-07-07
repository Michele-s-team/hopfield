#!/usr/bin/env python3

import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator

mpl.rcParams.update({
    "text.usetex"            : True,
    "font.family"            : "serif",
    "font.serif"             : ["Computer Modern Roman"],
    "axes.formatter.use_mathtext": True,

    "xtick.direction"        : "in",
    "ytick.direction"        : "in",
    "xtick.top"              : True,
    "ytick.right"            : True,
    "xtick.major.size"       : 5,
    "ytick.major.size"       : 5,
    "xtick.minor.size"       : 3,
    "ytick.minor.size"       : 3,
    "xtick.major.width"      : 0.8,
    "ytick.major.width"      : 0.8,
    "xtick.minor.visible"    : True,
    "ytick.minor.visible"    : True,
    "axes.linewidth"         : 0.8,
    "axes.spines.top"        : True,
    "axes.spines.right"      : True,

    "legend.frameon"         : False,
    "legend.fontsize"        : 8,
    "legend.handlelength"    : 2.0,
    "legend.labelspacing"    : 0.3,
    "legend.handletextpad"   : 0.5,

    "font.size"              : 10,
    "axes.labelsize"         : 11,
    "xtick.labelsize"        : 9,
    "ytick.labelsize"        : 9,
    "lines.linewidth"        : 1.2,

    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

# ── Palette : marqueurs creux, couleurs vives — même charte que le script Ising ──
COLORS  = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#e07b00', '#8c564b', '#17becf']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*']
# fillstyle='none' → marqueurs creux

MS  = 5.0
EW  = 0.6
CS  = 2


# ============================================================
# UTIL
# ============================================================

def extract_r(name):
    m = re.search(r"_r(\d+)\.csv$", name)
    return int(m.group(1)) if m else None


def extract_N(name: str):
    m = re.search(r"N(\d+)", name)
    return int(m.group(1)) if m else None


# ============================================================
# LOAD OVERLAPS FROM PRECOMPUTED CSV
# ============================================================

def load_overlaps(overlap_file):
    df = pd.read_csv(overlap_file)
    sweeps = df["sweep"].to_numpy(dtype=int)
    m_cols = [c for c in df.columns if c.startswith("m_")]
    overlaps = df[m_cols].to_numpy(dtype=float)
    return sweeps, overlaps


# ============================================================
# PROCESS ONE ALPHA
# Path structure: beta*/N*/alpha_*/overlaps/overlaps_r*.csv
# ============================================================

def process_alpha(alpha_dir):

    overlaps_dir = alpha_dir / "overlaps"
    if not overlaps_dir.exists():
        return None, None

    overlap_files = sorted(overlaps_dir.glob("overlaps_r*.csv"))
    if not overlap_files:
        return None, None

    final_values = []

    for ofile in overlap_files:
        _, ov = load_overlaps(ofile)

        # mu* = max overlap (in absolute value) at final sweep
        mu_star_curve = np.max(np.abs(ov), axis=1)
        final_values.append(mu_star_curve[-1])

    final_values = np.array(final_values)

    mean = final_values.mean()
    err  = final_values.std(ddof=1) / np.sqrt(len(final_values)) if len(final_values) > 1 else 0.0

    return mean, err


# ============================================================
# PROCESS ONE N (loops over all alpha_* subfolders)
# ============================================================

def process_N(N_dir):

    ALL_ALPHA_DIRS = sorted(
        N_dir.glob("alpha_*"),
        key=lambda p: float(p.name.split("_")[1])
    )
    if not ALL_ALPHA_DIRS:
        return None, None, None

    alphas, means, errors = [], [], []

    for alpha_dir in ALL_ALPHA_DIRS:
        alpha = float(alpha_dir.name.split("_")[1])
        print(f"   processing alpha = {alpha}")
        mean, err = process_alpha(alpha_dir)
        if mean is None:
            continue
        alphas.append(alpha)
        means.append(mean)
        errors.append(err)

    if not alphas:
        return None, None, None

    order = np.argsort(alphas)
    alphas = np.array(alphas)[order]
    means  = np.array(means)[order]
    errors = np.array(errors)[order]

    return alphas, means, errors


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # PATH STRUCTURE: beta{beta}/N{N}/alpha_*/overlaps/
    # ============================================================

    root_dir = Path("../results/Hopfield/phase_transition_multiN_init_from_pattern/")

    print(f"[INFO] cwd      : {Path.cwd()}")
    print(f"[INFO] root_dir : {root_dir.resolve()}")

    # ── Select beta directory ──────────────────────────────
    ALL_BETA_DIRS = sorted(
        [d for d in root_dir.glob("beta*") if d.is_dir()],
        key=lambda p: float(p.name.replace("beta", ""))
    )
    if not ALL_BETA_DIRS:
        raise ValueError(f"No beta* directories found in {root_dir}")

    BETA_SELECTION = None  # e.g. 5.0 to force a specific beta, else first found
    if BETA_SELECTION is None:
        beta_dir = ALL_BETA_DIRS[0]
    else:
        beta_dir = next(
            d for d in ALL_BETA_DIRS
            if abs(float(d.name.replace("beta", "")) - BETA_SELECTION) < 1e-9
        )
    beta = float(beta_dir.name.replace("beta", ""))

    print(f"\n[INFO] {len(ALL_BETA_DIRS)} beta folder(s) found, using: {beta_dir.name}")

    # Figures now saved directly inside the chosen beta folder
    output_dir = beta_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Select N directories: ALL of them ──────────────────
    ALL_N_DIRS = sorted(
        [d for d in beta_dir.glob("N*") if d.is_dir()],
        key=lambda p: extract_N(p.name)
    )
    if not ALL_N_DIRS:
        raise ValueError(f"No N* directories found in {beta_dir}")

    print(f"[INFO] {len(ALL_N_DIRS)} N folder(s) found, plotting all of them:")
    for d in ALL_N_DIRS:
        print("  ", d.name)

    alpha_c = 0.12256598472304177

    # ============================================================
    # CSV CACHE: N / alpha / mean / err, stored directly in the beta folder
    # ============================================================

    csv_path = beta_dir / "phase_transition_data.csv"

    results = {}  # N -> (alphas, means, errors)

    if csv_path.exists():
        print(f"\n[CACHE] Found existing data file: {csv_path}")
        print("[CACHE] Loading values directly instead of recomputing.")

        df_cache = pd.read_csv(csv_path)
        for N, group in df_cache.groupby("N"):
            group = group.sort_values("alpha")
            results[int(N)] = (
                group["alpha"].to_numpy(dtype=float),
                group["mean_overlap"].to_numpy(dtype=float),
                group["error"].to_numpy(dtype=float),
            )

    else:
        print(f"\n[CACHE] No data file found at {csv_path}")
        print("[CACHE] Computing values from raw overlap files...")

        # ============================================================
        # COMPUTE mu* PER (N, ALPHA)
        # ============================================================

        for N_dir in ALL_N_DIRS:
            N = extract_N(N_dir.name)
            print(f"processing N{N}")
            alphas, means, errors = process_N(N_dir)
            if alphas is None:
                print(f"  [SKIP] {N_dir.name}: no usable data")
                continue
            results[N] = (alphas, means, errors)

        if not results:
            raise ValueError("No data found for any N.")

        # ── Save to CSV for reuse next time ──
        rows = []
        for N, (alphas, means, errors) in results.items():
            for a, m, e in zip(alphas, means, errors):
                rows.append({"N": N, "alpha": a, "mean_overlap": m, "error": e})
        df_out = pd.DataFrame(rows).sort_values(["N", "alpha"])
        df_out.to_csv(csv_path, index=False)
        print(f"[CACHE] Saved data to: {csv_path}")

    if not results:
        raise ValueError("No data found for any N.")

    # ── Assignation couleur/marqueur par N — même logique que le script Ising ──
    Ns_sorted = sorted(results.keys())
    n = len(Ns_sorted)
    colors  = {N: COLORS[n - 1 - Ns_sorted.index(N)]  for N in Ns_sorted}
    markers = {N: MARKERS[n - 1 - Ns_sorted.index(N)] for N in Ns_sorted}

    def plot_series(ax, N, alphas, y_vals, err_vals):
        c  = colors[N]
        mk = markers[N]
        ax.plot(alphas, y_vals,
                color=c, lw=1.2, zorder=2)
        ax.errorbar(alphas, y_vals, yerr=err_vals,
                    fmt=mk, color=c,
                    ms=MS, lw=0,
                    elinewidth=EW, capsize=CS, capthick=EW,
                    fillstyle="none",
                    markeredgewidth=1.0,
                    zorder=4,
                    label=fr"$N={N}$")

    # ============================================================
    # PLOT 1: OVERLAP MAX vs ALPHA (all N)
    # ============================================================

    fig1, ax1 = plt.subplots(figsize=(3.4, 2.8))

    for N in Ns_sorted:
        alphas, means, errors = results[N]
        plot_series(ax1, N, alphas, means, errors)

    ax1.axvline(alpha_c, color="0.4", lw=0.8, ls="--", zorder=0)

    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks([alpha_c])
    ax1_top.set_xticklabels([r"$\alpha_c$"], fontsize=8)
    ax1_top.tick_params(direction="in", length=5, width=0.8)
    ax1_top.xaxis.set_minor_locator(ticker.NullLocator())

    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.set_ylim(0.2, 1.05)
    ax1_top.set_xlim(ax1.get_xlim())
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(r"$\langle m^*\rangle$")

    ax1.legend(loc="center left", ncol=1, columnspacing=0.8, fontsize=8)

    plt.tight_layout(pad=0.3)
    fig1.savefig(output_dir / "overlap_max_vs_alpha.pdf", bbox_inches="tight")
    fig1.savefig(output_dir / "overlap_max_vs_alpha.png", bbox_inches="tight")
    plt.close(fig1)

    # ============================================================
    # PLOT 2: Reconstruction error vs alpha (all N)
    # ============================================================

    fig2, ax2 = plt.subplots(figsize=(3.4, 2.8))

    for N in Ns_sorted:
        alphas, means, errors = results[N]
        reconstruction_error = (1 - means) / 2
        reconstruction_error_err = errors / 2
        plot_series(ax2, N, alphas, reconstruction_error, reconstruction_error_err)

    ax2.axvline(alpha_c, color="0.4", lw=0.8, ls="--", zorder=0)

    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks([alpha_c])
    ax2_top.set_xticklabels([r"$\alpha_c$"], fontsize=8)
    ax2_top.tick_params(direction="in", length=5, width=0.8)
    ax2_top.xaxis.set_minor_locator(ticker.NullLocator())

    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.set_ylim(-0.03, 0.42)
    ax2_top.set_xlim(ax2.get_xlim())
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel("Reconstruction error")

    ax2.legend(loc="center left", ncol=1, columnspacing=0.8, fontsize=8)

    plt.tight_layout(pad=0.3)
    fig2.savefig(output_dir / "reconstruction_error_vs_alpha.pdf", bbox_inches="tight")
    fig2.savefig(output_dir / "reconstruction_error_vs_alpha.png", bbox_inches="tight")
    plt.close(fig2)

    print(f"\nSaved figures to: {output_dir.resolve()}")