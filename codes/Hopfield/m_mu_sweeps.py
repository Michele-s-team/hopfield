#!/usr/bin/env python3

import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import mplcursors


# ============================================================
# UTIL
# ============================================================

def extract_r(name):
    m = re.search(r"_r(\d+)\.csv$", name)
    return int(m.group(1)) if m else None

def extract_N(name: str):
    m = re.search(r"N(\d+)", name)
    return int(m.group(1)) if m else None


def fix_sweeps(sweeps):
    sweeps_plot = sweeps.copy().astype(float)
    if sweeps_plot[0] == 0 and len(sweeps_plot) > 1:
        sweeps_plot[0] = sweeps_plot[1] / 2.0
    return sweeps_plot


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
# ============================================================

def process_alpha(alpha_dir, overlaps_dir_name):
    m = re.search(r"overlaps_N(\d+)_beta([0-9.]+)", overlaps_dir_name)
    N    = int(m.group(1))
    beta = float(m.group(2))

    overlaps_dir = alpha_dir / overlaps_dir_name
    print(f"  [DEBUG] looking for {overlaps_dir}")
    print(f"  [DEBUG] exists: {overlaps_dir.exists()}")
    if not overlaps_dir.exists():
        print(f"  [WARN] pas de dossier overlaps dans {alpha_dir.name}")
        return None, None, None

    overlap_files = sorted(overlaps_dir.glob(f"overlaps_N{N}_beta{beta}_r*.csv"))
    print(f"  [DEBUG] files found: {len(overlap_files)}")
    if not overlap_files:
        return None, None, None

    curves = []
    sweeps = None

    for j, ofile in enumerate(overlap_files, start=1):
        print(f"    realization {j}/{len(overlap_files)}", end="\r", flush=True)
        sw, ov = load_overlaps(ofile)
        curve = np.max(np.abs(ov), axis=1)
        curves.append(curve)
        if sweeps is None:
            sweeps = sw

    print()

    min_len = min(len(c) for c in curves)
    curves = np.array([c[:min_len] for c in curves])

    mean_curve = curves.mean(axis=0)
    std_curve  = curves.std(axis=0) / np.sqrt(len(curves))

    return sweeps[:min_len], mean_curve, std_curve


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    root_dir = Path("../../results")
    print(root_dir.resolve())

    ALL_ALPHA_DIRS = sorted(root_dir.glob("alpha_*"))

    ALPHA_SELECTION = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    alpha_dirs = [
        ALL_ALPHA_DIRS[i - 1]
        for i in ALPHA_SELECTION
        if 1 <= i <= len(ALL_ALPHA_DIRS)
    ]

    print("\nSelected alpha folders:")
    for d in alpha_dirs:
        print(" ", d.name)

    # détecter le nom exact du dossier overlaps depuis le premier alpha
    first_alpha = alpha_dirs[0]
    overlap_dirs = sorted(first_alpha.glob("overlaps_N*_beta*"))
    if not overlap_dirs:
        raise RuntimeError("No overlaps_N*_beta* folder found")

    overlaps_dir_name = overlap_dirs[0].name   # nom exact, ex: overlaps_N1024_beta5.000000

    m = re.search(r"overlaps_N(\d+)_beta([0-9.]+)", overlaps_dir_name)
    N_label    = m.group(1)
    beta_label = m.group(2)
    N    = int(N_label)
    beta = float(beta_label)

    print(f"\nDetected: N={N}, beta={beta}, overlaps folder='{overlaps_dir_name}'")

    # ----------------------------------------------------------
    # Plot 1 : max_mu |m^mu| moyenné sur les réalisations
    # ----------------------------------------------------------

    plt.figure(figsize=(8, 5))
    sweeps_plot_last = None
    results = {}

    for i, alpha_dir in enumerate(alpha_dirs, start=1):
        alpha = float(alpha_dir.name.split("_")[1])
        print(f"[{i}/{len(alpha_dirs)}] processing {alpha_dir.name}")

        sweeps, mean_curve, std_curve = process_alpha(alpha_dir, overlaps_dir_name)
        if sweeps is None:
            print(f"  [WARN] no data for {alpha_dir.name}")
            continue

        results[alpha] = (sweeps, mean_curve, std_curve)

        sweeps_plot = fix_sweeps(sweeps)
        sweeps_plot_last = sweeps_plot

        plt.plot(sweeps_plot, mean_curve, label=f"{alpha:.3f}")
        plt.fill_between(
            sweeps_plot,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
        )

    plt.xlabel("sweep")
    plt.ylabel(r"$\langle \max_\mu |m^\mu| \rangle$")
    plt.xscale("log")
    if sweeps_plot_last is not None:
        plt.xlim([sweeps_plot_last[0] * 0.95, sweeps_plot_last[-1] * 1.05])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if not results:
        raise RuntimeError("No results to export — check overlaps folder names.")

    dfs = []
    for alpha, (sweeps, mean_curve, std_curve) in results.items():
        dfs.append(pd.DataFrame({
            "sweep":             sweeps,
            f"mean_{alpha:.3f}": mean_curve,
            f"sem_{alpha:.3f}":  std_curve,
        }).set_index("sweep"))

    df_export = pd.concat(dfs, axis=1).reset_index()
    df_export.to_csv(
        f"../../results/max_overlap_vs_sweeps_N{N_label}_beta{beta_label}.csv",
        index=False
    )
    print(f"Exported max_overlap_vs_sweeps_N{N_label}_beta{beta_label}.csv")

    # ----------------------------------------------------------
    # Plots 2 & 3 : zoom sur un alpha particulier
    # ----------------------------------------------------------

    n_alpha = 2   # index 0-based dans ALL_ALPHA_DIRS
    alpha_dir   = ALL_ALPHA_DIRS[n_alpha]
    overlaps_dir = alpha_dir / overlaps_dir_name

    alpha = float(alpha_dir.name.split("_")[1])
    print(f"\nalpha = {alpha:.3f}")

    # ----------------------------------------------------------
    # Plot 2 : tous les m^mu pour une réalisation donnée
    # ----------------------------------------------------------

    r_target = 16

    ofile = overlaps_dir / f"overlaps_N{N}_beta{beta}_r{r_target}.csv"
    sweeps, overlaps = load_overlaps(ofile)

    sweeps_plot = fix_sweeps(sweeps)
    n_patterns  = overlaps.shape[1]
    dominant    = np.argmax(np.abs(overlaps[-1, :]))

    plt.figure(figsize=(9, 5))

    for mu in range(n_patterns):
        lw  = 2.0 if mu == dominant else 0.8
        zo  = 3   if mu == dominant else 1
        lbl = rf"$m^{{{mu}}}$" + (" ★" if mu == dominant else "")
        plt.plot(sweeps_plot, overlaps[:, mu], lw=lw, zorder=zo, label=lbl)

    plt.axhline(0, color="k", lw=0.5, ls="--")
    plt.xlabel("sweep")
    plt.ylabel(r"$m^\mu$")
    plt.title(rf"$\alpha = {alpha:.3f}$, $N = {N_label}$, réalisation $r = {r_target}$")
    plt.xscale("log")
    plt.xlim([sweeps_plot[0] * 0.95, sweeps_plot[-1] * 1.05])
    plt.grid(True, alpha=0.4)
    plt.legend(ncol=4, fontsize=7)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------
    # Plot 3 : évolution de m^mu* pour toutes les réalisations
    # ----------------------------------------------------------

    plt.figure(figsize=(9, 5))

    ofiles   = sorted(overlaps_dir.glob(f"overlaps_N{N}_beta{beta}_r*.csv"))
    n_real   = len(ofiles)
    colors   = [hsv_to_rgb([i / n_real, 0.55, 0.75]) for i in range(n_real)]

    lines = []

    for idx, ofile in enumerate(ofiles):
        r = extract_r(ofile.name)
        sweeps, ov = load_overlaps(ofile)
        mu_star     = np.argmax(np.abs(ov[-1]))
        sweeps_plot = fix_sweeps(sweeps)

        line, = plt.plot(sweeps_plot, ov[:, mu_star],
                         lw=0.8, alpha=0.6, color=colors[idx])
        lines.append((line, r))

    cursor = mplcursors.cursor([l for l, _ in lines], hover=True)

    @cursor.connect("add")
    def on_add(sel):
        r = next(idx for l, idx in lines if l is sel.artist)
        sel.annotation.set_text(f"realization r={r}")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

    plt.axhline(0, color="k", lw=0.5, ls="--")
    plt.xlabel("sweep")
    plt.xscale("log")
    plt.ylabel(r"$m^{\mu^*}$")
    plt.title(rf"$\alpha = {alpha:.3f}$, $N = {N_label}$ — toutes les réalisations")
    plt.grid(True, alpha=0.4)
    plt.xlim([sweeps_plot[0] * 0.95, sweeps_plot[-1] * 1.05])
    plt.tight_layout()
    plt.show()