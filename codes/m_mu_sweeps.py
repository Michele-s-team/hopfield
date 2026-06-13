#!/usr/bin/env python3

import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


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

    config_df = pd.read_csv(config_file, dtype=str)
    pattern_df = pd.read_csv(pattern_file, dtype=str)

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
            df = pd.read_csv(cfile, dtype=str)
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

    N_ALPHA = 7
    alpha_dirs = sorted(root_dir.glob("alpha_*"))[5:5+N_ALPHA]

    plt.figure(figsize=(8, 5))

    for alpha_dir in alpha_dirs:

        alpha = float(alpha_dir.name.split("_")[1])
        print(alpha)

        sweeps, mean_curve, std_curve = process_alpha(alpha_dir)

        if sweeps is None:
            continue

        plt.plot(sweeps, mean_curve, label=f"{alpha:.3f}")
        plt.fill_between(
            sweeps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2
        )

    plt.xlabel("sweep")
    plt.ylabel(r"$\langle \max_\mu |m^\mu| \rangle$")
    plt.xscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()