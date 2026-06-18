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


def extract_N(name):
    m = re.search(r"N(\d+)", name)
    return int(m.group(1)) if m else None


def num_blocks(N):
    return (N + 63) // 64


def fix_sweeps(sweeps):
    """Remplace le sweep 0 par sweeps[1]/2 pour permettre xscale log."""
    sweeps_plot = sweeps.copy().astype(float)
    if sweeps_plot[0] == 0 and len(sweeps_plot) > 1:
        sweeps_plot[0] = sweeps_plot[1] / 2
    return sweeps_plot


# ============================================================
# OVERLAP
# ============================================================

def overlap_binary(row_cfg, row_pat, N):
    n_full = N // 64
    remainder = N % 64

    corr = 0

    for b in range(n_full):
        s = int(str(row_cfg[f"block{b}"]))
        p = int(str(row_pat[f"block{b}"]))
        xor = s ^ p
        corr += 64 - 2 * xor.bit_count()

    if remainder:
        s = int(str(row_cfg[f"block{n_full}"]))
        p = int(str(row_pat[f"block{n_full}"]))
        mask = (1 << remainder) - 1
        xor = (s ^ p) & mask
        corr += remainder - 2 * xor.bit_count()

    return corr / N


# ============================================================
# PROCESS ONE REALIZATION
# ============================================================

def compute_curve(config_file, pattern_file, N):

    config_df = pd.read_csv(config_file, dtype=str).dropna()
    pattern_df = pd.read_csv(pattern_file, dtype=str).dropna()

    n_sweeps = len(config_df)
    curve = np.zeros(n_sweeps)

    for i in range(n_sweeps):
        row_cfg = config_df.iloc[i]

        m_max = -1.0

        for _, row_pat in pattern_df.iterrows():
            m = abs(overlap_binary(row_cfg, row_pat, N))
            if m > m_max:
                m_max = m

        curve[i] = m_max

    return curve


def compute_all_overlaps(config_file, pattern_file, N):
    """Retourne un array (n_sweeps, n_patterns) des overlaps m^mu."""
    config_df = pd.read_csv(config_file, dtype=str).dropna()
    pattern_df = pd.read_csv(pattern_file, dtype=str).dropna()

    n_sweeps = len(config_df)
    n_patterns = len(pattern_df)
    overlaps = np.zeros((n_sweeps, n_patterns))

    for i, row_cfg in config_df.iterrows():
        for mu, row_pat in pattern_df.iterrows():
            overlaps[i, mu] = overlap_binary(row_cfg, row_pat, N)

    return overlaps


# ============================================================
# PROCESS ONE ALPHA
# ============================================================

def process_alpha(alpha_dir, N=None):

    spins_dir = alpha_dir / "spins"
    patterns_dir = alpha_dir / "patterns"

    config_files = sorted(spins_dir.glob("*.csv"))

    pattern_map = {}
    for p in patterns_dir.glob("*.csv"):
        r = extract_r(p.name)
        if r is not None:
            pattern_map[r] = p

    if N is None:
        any_file = next(iter(patterns_dir.glob("*.csv")))
        N = extract_N(any_file.name)

    curves = []
    sweeps = None

    for cfile in config_files:

        r = extract_r(cfile.name)
        if r is None or r not in pattern_map:
            continue

        pfile = pattern_map[r]

        curve = compute_curve(cfile, pfile, N)
        curves.append(curve)

        if sweeps is None:
            df = pd.read_csv(cfile, dtype=str).dropna()
            sweeps = df["sweep"].to_numpy()

    if not curves:
        return None, None, None

    min_len = min(len(c) for c in curves)
    curves = np.array([c[:min_len] for c in curves])

    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0) / np.sqrt(len(curves))

    return sweeps[:min_len], mean_curve, std_curve


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    root_dir = Path("../results")

    N_ALPHA = 3
    alpha_dirs = sorted(root_dir.glob("alpha_*"))[:N_ALPHA]

    # ----------------------------------------------------------
    # Plot 1 : max_mu |m^mu| moyenné sur les réalisations
    # ----------------------------------------------------------

    plt.figure(figsize=(8, 5))

    for alpha_dir in alpha_dirs:

        alpha = float(alpha_dir.name.split("_")[1])
        print(alpha)

        sweeps, mean_curve, std_curve = process_alpha(alpha_dir)

        if sweeps is None:
            continue

        sweeps_plot = fix_sweeps(sweeps)

        plt.plot(sweeps_plot, mean_curve, label=f"{alpha:.3f}")
        plt.fill_between(
            sweeps_plot,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2
        )

    plt.xlabel("sweep")
    plt.ylabel(r"$\langle \max_\mu |m^\mu| \rangle$")
    plt.xscale("log")
    plt.xlim([sweeps_plot[0]*0.95,sweeps_plot[-1]*1.05])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    alpha_dir = sorted(root_dir.glob("alpha_*"))[9]
    alpha = float(alpha_dir.name.split("_")[1])
    print(f"alpha = {alpha:.3f}")

    spins_dir = alpha_dir / "spins"
    patterns_dir = alpha_dir / "patterns"

    # ----------------------------------------------------------
    # Plot 2 : tous les m^mu pour une réalisation donnée
    # ----------------------------------------------------------

    r_target = 16

    config_file = next(
        f for f in sorted(spins_dir.glob("*.csv")) if extract_r(f.name) == r_target
    )
    pattern_file = next(
        p for p in patterns_dir.glob("*.csv") if extract_r(p.name) == r_target
    )

    N = extract_N(pattern_file.name)
    print(f"N = {N}, realization r = {r_target}")

    overlaps = compute_all_overlaps(config_file, pattern_file, N)

    config_df = pd.read_csv(config_file, dtype=str).dropna()
    sweeps = config_df["sweep"].to_numpy(dtype=float)
    sweeps_plot = fix_sweeps(sweeps)

    n_patterns = overlaps.shape[1]

    plt.figure(figsize=(9, 5))

    dominant = np.argmax(np.abs(overlaps[-1, :]))

    for mu in range(n_patterns):
        lw = 2.0 if mu == dominant else 0.8
        zo = 3 if mu == dominant else 1
        lbl = rf"$m^{{{mu}}}$" + (" ★" if mu == dominant else "")
        plt.plot(sweeps_plot, overlaps[:, mu], lw=lw, zorder=zo, label=lbl)

    plt.axhline(0, color="k", lw=0.5, ls="--")
    plt.xlabel("sweep")
    plt.ylabel(r"$m^\mu$")
    plt.title(rf"$\alpha = {alpha:.3f}$, $N = {N}$, réalisation $r = {r_target}$")
    plt.xscale("log")
    plt.xlim([sweeps_plot[0]*0.95,sweeps_plot[-1]*1.05])
    plt.grid(True, alpha=0.4)
    plt.legend(ncol=4, fontsize=7)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------
    # Plot 3 : évolution de m^mu* pour toutes les réalisations
    #          (mu* = pattern dominant à la fin de chaque réalisation)
    # ----------------------------------------------------------

    N = extract_N(pattern_file.name)

    pattern_map = {}
    for p in patterns_dir.glob("*.csv"):
        r = extract_r(p.name)
        if r is not None:
            pattern_map[r] = p

    plt.figure(figsize=(9, 5))

    config_files = sorted(spins_dir.glob("*.csv"))
    n_real = len(config_files)
    colors = [hsv_to_rgb([i / n_real, 0.55, 0.75]) for i in range(n_real)]

    lines = []

    for idx, cfile in enumerate(config_files):
        r = extract_r(cfile.name)
        if r is None or r not in pattern_map:
            continue

        overlaps = compute_all_overlaps(cfile, pattern_map[r], N)
        mu_star = np.argmax(np.abs(overlaps[-1]))

        config_df = pd.read_csv(cfile, dtype=str).dropna()
        sweeps = config_df["sweep"].to_numpy(dtype=float)
        sweeps_plot = fix_sweeps(sweeps)

        line, = plt.plot(sweeps_plot, overlaps[:, mu_star],
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
    plt.title(rf"$\alpha = {alpha:.3f}$, $N = {N}$ — toutes les réalisations")
    plt.grid(True, alpha=0.4)
    plt.xlim([sweeps_plot[0]*0.95,sweeps_plot[-1]*1.05])
    plt.tight_layout()
    plt.show()