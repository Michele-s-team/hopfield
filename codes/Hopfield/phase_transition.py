#!/usr/bin/env python3

import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import mplcursors
import os
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator

mpl.rcParams.update({
    # LaTeX
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],

    # Fonts
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    # Axes
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,

    # Ticks
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

    # Lines
    "lines.linewidth": 1.4,
    "lines.markersize": 5,

    # Legend
    "legend.frameon": False,
    "legend.handlelength": 2.0,

    # Savefig
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


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
# FIND BETA DIRECTORY (handles different beta formats)
# ============================================================

def find_beta_dir(parent_dir, beta):
    """
    Find the beta directory regardless of formatting (beta5.0, beta5.000000, etc.)
    """
    # Try exact match first
    exact_path = parent_dir / f"beta{beta}"
    if exact_path.exists() and exact_path.is_dir():
        return exact_path
    
    # Try with different decimal formats
    beta_strs = [
        f"beta{beta:.6f}",
        f"beta{beta:.5f}",
        f"beta{beta:.4f}",
        f"beta{beta:.3f}",
        f"beta{beta:.2f}",
        f"beta{beta:.1f}",
        f"beta{beta:.0f}",
        f"beta{beta}",
    ]
    
    # Remove duplicates while preserving order
    beta_strs = list(dict.fromkeys(beta_strs))
    
    for beta_str in beta_strs:
        path = parent_dir / beta_str
        if path.exists() and path.is_dir():
            return path
    
    # If nothing found, try to find any beta* directory
    beta_dirs = sorted([d for d in parent_dir.glob("beta*") if d.is_dir()])
    if beta_dirs:
        # Try to parse the beta value from the directory name
        for d in beta_dirs:
            try:
                match = re.search(r"beta([\d.]+)", d.name)
                if match:
                    found_beta = float(match.group(1))
                    if abs(found_beta - beta) < 1e-6:
                        return d
            except:
                continue
        
        # If we can't match, return the first one as a guess
        print(f"  Could not find exact beta={beta}, using {beta_dirs[0].name} as fallback")
        return beta_dirs[0]
    
    return None

# ============================================================
# FIND OVERLAPS DIRECTORY
# ============================================================

def find_overlaps_dir(alpha_dir, N, beta):
    """
    Find the overlaps directory with the correct path structure.
    Path: alpha/N/beta/overlaps/
    """
    # Build the path: alpha/N/beta/overlaps/
    N_dir = alpha_dir / f"N{N}"
    if not N_dir.exists():
        return None
    
    beta_dir = find_beta_dir(N_dir, beta)
    if beta_dir is None:
        return None
    
    # Try overlaps directory
    overlaps_dir = beta_dir / "overlaps"
    if overlaps_dir.exists() and overlaps_dir.is_dir():
        return overlaps_dir
    
    return None

# ============================================================
# PROCESS ONE ALPHA
# ============================================================

def process_alpha(alpha_dir, N, beta):

    overlaps_dir = find_overlaps_dir(alpha_dir, N, beta)
    if overlaps_dir is None:
        return None, None

    overlap_files = sorted(overlaps_dir.glob("overlaps_r*.csv"))
    if not overlap_files:
        return None, None

    final_values = []
    sweeps = None

    for ofile in overlap_files:
        sw, ov = load_overlaps(ofile)

        # mu* = max overlap (in absolute value) at final sweep
        mu_star_curve = np.max(np.abs(ov), axis=1)

        final_values.append(mu_star_curve[-1])

        if sweeps is None:
            sweeps = sw

    final_values = np.array(final_values)

    mean = final_values.mean()
    err  = final_values.std(ddof=1) / np.sqrt(len(final_values))

    return mean, err

# ============================================================
# GET OVERLAP FILES (helper function)
# ============================================================

def get_overlap_files(overlaps_dir, N, beta):
    """
    Get all overlap files, trying both new and old naming conventions.
    """
    # New format: overlaps_r*.csv
    files = sorted(overlaps_dir.glob("overlaps_r*.csv"))
    
    # Fallback: old format
    if not files:
        files = sorted(overlaps_dir.glob(f"overlaps_N{N}_beta*_r*.csv"))
    
    # Final fallback
    if not files:
        files = sorted(overlaps_dir.glob("*overlaps*.csv"))
    
    return files























# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # PATH STRUCTURE: alpha_*/N1024/beta5.000000/overlaps/
    # ============================================================
    
    root_dir = Path("../results/Hopfield/phase_transition/init_from_pattern/")
    output_dir = root_dir / "figures"
    output_dir.mkdir(exist_ok=True)

    # Get all alpha directories
    ALL_ALPHA_DIRS = sorted(root_dir.glob("alpha_*"))

    ALPHA_SELECTION = None

    if ALPHA_SELECTION is None:
        alpha_dirs = ALL_ALPHA_DIRS
    else:
        alpha_dirs = [
            ALL_ALPHA_DIRS[i - 1]
            for i in ALPHA_SELECTION
            if 1 <= i <= len(ALL_ALPHA_DIRS)
        ]

    print("\nSelected alpha folders:")
    for d in alpha_dirs:
        print(" ", d.name)

    # ============================================================
    # DETECT N AND BETA FROM THE FIRST ALPHA DIRECTORY
    # ============================================================
    
    first_alpha = alpha_dirs[0]
    
    # Look for N directories inside the first alpha
    N_dirs = sorted([d for d in first_alpha.glob("N*") if d.is_dir()])
    if not N_dirs:
        raise ValueError(f"No N directories found in {first_alpha}")
    
    first_N_dir = N_dirs[0]
    N_label = first_N_dir.name.replace("N", "")
    N = int(N_label)
    
    # Look for beta directories inside the first N directory
    beta_dirs = sorted([d for d in first_N_dir.glob("beta*") if d.is_dir()])
    if not beta_dirs:
        raise ValueError(f"No beta directories found in {first_N_dir}")
    
    first_beta_dir = beta_dirs[0]
    beta_label = first_beta_dir.name.replace("beta", "")
    beta = float(beta_label)

    print(f"\nDetected: N={N}, beta={beta}")
    print(f"Beta folder format: {first_beta_dir.name}")
    print(f"Path structure: alpha_*/N{N}/beta*/overlaps/")

    # ============================================================
    # DEBUG: Show the actual path for first alpha
    # ============================================================
    
    print("\n" + "="*60)
    print("DEBUG: Checking path structure")
    print("="*60)
    test_alpha = alpha_dirs[0]
    print(f"Testing with: {test_alpha}")
    test_N_dir = test_alpha / f"N{N}"
    print(f"  N dir exists: {test_N_dir.exists()} -> {test_N_dir}")
    if test_N_dir.exists():
        test_beta_dir = find_beta_dir(test_N_dir, beta)
        print(f"  Beta dir found: {test_beta_dir}")
        if test_beta_dir:
            test_overlaps = test_beta_dir / "overlaps"
            print(f"  Overlaps dir exists: {test_overlaps.exists()} -> {test_overlaps}")
            if test_overlaps.exists():
                test_files = get_overlap_files(test_overlaps, N, beta)
                print(f"  Found {len(test_files)} overlap files")
                if test_files:
                    print(f"  Example: {test_files[0].name}")
    print("="*60 + "\n")

    results = {}

    alphas = []
    means = []
    errors = []

    for alpha_dir in alpha_dirs:

        alpha = float(alpha_dir.name.split("_")[1])
        print(alpha)

        mean, err = process_alpha(alpha_dir, N, beta)
        if mean is None:
            continue

        alphas.append(alpha)
        means.append(mean)
        errors.append(err)

# ============================================================
# SORT RESULTS (important for clean plots)
# ============================================================

order = np.argsort(alphas)
alphas = np.array(alphas)[order]
means  = np.array(means)[order]
errors = np.array(errors)[order]

<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
alpha_c = 0.138

=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
# ============================================================
# PLOT 1: OVERLAP MAX vs ALPHA
# ============================================================

plt.figure(figsize=(4.5, 3.5))

plt.errorbar(
    alphas,
    means,
    yerr=errors,
    fmt="o-",
    capsize=3,
)
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\langle m^*\rangle$")
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
plt.ylim(-0.05, 1.05)
plt.grid(alpha=0.3)

plt.axvline(alpha_c, linestyle="--", color="black", linewidth=1.0, zorder=0)
ax = plt.gca()
secax = ax.secondary_xaxis('top')
secax.set_xticks([alpha_c])
secax.set_xticklabels([r"$\alpha_c$"])
secax.tick_params(direction="in", length=4)

=======
plt.ylim(-0.01, 1.01)
plt.grid(alpha=0.3)

>>>>>>> Stashed changes
=======
plt.ylim(-0.01, 1.01)
plt.grid(alpha=0.3)

>>>>>>> Stashed changes
=======
plt.ylim(-0.01, 1.01)
plt.grid(alpha=0.3)

>>>>>>> Stashed changes
plt.tight_layout()
plt.savefig(output_dir / "overlap_max_vs_alpha.pdf")
plt.savefig(output_dir / "overlap_max_vs_alpha.png")
plt.close()

# ============================================================
# PLOT 2: Reconstruction error vs alpha
# ============================================================

reconstruction_error = (1 - means) / 2
reconstruction_error_err = errors / 2

plt.figure(figsize=(4.5, 3.5))

plt.errorbar(
    alphas,
    reconstruction_error,
    yerr=reconstruction_error_err,
    fmt="o-",
    capsize=3,
)

plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
plt.xlabel(r"$\alpha$")
plt.ylabel("Reconstruction error")
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
plt.ylim(-0.05, 0.55)
plt.grid(alpha=0.3)

plt.axvline(alpha_c, linestyle="--", color="black", linewidth=1.0, zorder=0)
ax = plt.gca()
secax = ax.secondary_xaxis('top')
secax.set_xticks([alpha_c])
secax.set_xticklabels([r"$\alpha_c$"])
secax.tick_params(direction="in", length=4)

=======
plt.ylim(-0.01, 0.55)
plt.grid(alpha=0.3)

>>>>>>> Stashed changes
=======
plt.ylim(-0.01, 0.55)
plt.grid(alpha=0.3)

>>>>>>> Stashed changes
=======
plt.ylim(-0.01, 0.55)
plt.grid(alpha=0.3)

>>>>>>> Stashed changes
plt.tight_layout()
plt.savefig(output_dir / "reconstruction_error_vs_alpha.pdf")
plt.savefig(output_dir / "reconstruction_error_vs_alpha.png")
plt.close()