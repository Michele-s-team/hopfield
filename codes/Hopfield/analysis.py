#!/usr/bin/env python3

import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import mplcursors
import os


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
    """
    Process all overlap files for a given alpha directory.
    Files are named: overlaps_r*.csv (without N and beta)
    """
    # Find the overlaps directory
    overlaps_dir = find_overlaps_dir(alpha_dir, N, beta)
    
    if overlaps_dir is None:
        return None, None, None

    # New format: overlaps_r*.csv
    overlap_files = sorted(overlaps_dir.glob("overlaps_r*.csv"))
    
    # Fallback: old format
    if not overlap_files:
        overlap_files = sorted(overlaps_dir.glob(f"overlaps_N{N}_beta*_r*.csv"))
    
    # Final fallback
    if not overlap_files:
        overlap_files = sorted(overlaps_dir.glob("*overlaps*.csv"))
    
    if not overlap_files:
        return None, None, None

    curves = []
    sweeps = None

    for ofile in overlap_files:
        sw, ov = load_overlaps(ofile)
        curve = np.max(np.abs(ov), axis=1)
        curves.append(curve)
        if sweeps is None:
            sweeps = sw

    min_len = min(len(c) for c in curves)
    curves = np.array([c[:min_len] for c in curves])

    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0) / np.sqrt(len(curves))

    return sweeps[:min_len], mean_curve, std_curve

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
    
    root_dir = Path("../results/Hopfield/")
    output_dir = root_dir / "figures"
    output_dir.mkdir(exist_ok=True)

    # Get all alpha directories
    ALL_ALPHA_DIRS = sorted(root_dir.glob("alpha_*"))

    ALPHA_SELECTION = list(range(1, 19))

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

    # ============================================================
    # FIGURE 1 : Mean overlap averaged over replicas for all alpha
    # ============================================================

    plt.figure(figsize=(8, 5))
    results = {}

    final_overlaps = []
    final_stds = []
    alphas_phase = []

    for i, alpha_dir in enumerate(alpha_dirs, start=1):

        alpha = float(alpha_dir.name.split("_")[1])

        sweeps, mean_curve, std_curve = process_alpha(alpha_dir, N, beta)
        if sweeps is None:
            print(f"  No data for alpha={alpha:.3f}, skipping...")
            continue

        print(f"  Processed alpha={alpha:.3f}")
        results[alpha] = (sweeps, mean_curve, std_curve)

        final_overlaps.append(mean_curve[-1])
        final_stds.append(std_curve[-1])
        alphas_phase.append(alpha)

        sweeps_plot = fix_sweeps(sweeps)

        plt.plot(sweeps_plot, mean_curve, label=f"$\\alpha = {alpha:.3f}$")
        plt.fill_between(sweeps_plot, mean_curve-std_curve, mean_curve+std_curve, alpha=0.2)

    plt.xscale("log")
    plt.xlabel("Number of sweeps (log scale)", fontsize=12)
    plt.ylabel(r"$\langle m^{\mu^*}  \rangle$", fontsize=12)
    plt.title(f"Mean overlap vs sweeps - N={N_label}, $\\beta={beta_label}$", fontsize=14)
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)

    fig1_path = output_dir / f"mean_overlap_N{N_label}_beta{beta_label}.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"FIGURE 1 (mean_overlap) saved: {fig1_path}")

    # ============================================================
    # FIGURE 2 : Phase diagram (final overlap vs alpha)
    # ============================================================

    plt.figure(figsize=(7,4))

    alphas_phase = np.array(alphas_phase)
    final_overlaps = np.array(final_overlaps)
    final_stds = np.array(final_stds)

    order = np.argsort(alphas_phase)
    plt.errorbar(alphas_phase[order], final_overlaps[order], yerr=final_stds[order], fmt="o-", 
                 capsize=3, capthick=1, markersize=6)

    plt.xlabel(r"$\alpha$ (storage capacity)", fontsize=12)
    plt.ylabel(r"$m^{\mu^*}$ (final)", fontsize=12)
    plt.title(f"Phase diagram - N={N_label}, $\\beta={beta_label}$", fontsize=14)
    plt.grid(True, alpha=0.3)

    fig2_path = output_dir / f"phase_diagram_N{N_label}_beta{beta_label}.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"FIGURE 2 (phase_diagram) saved: {fig2_path}")

    # ============================================================
    # FIGURE 3 : All mu components for a single run with mu* highlighted
    # ============================================================

    n_alpha = 2
    alpha_dir = ALL_ALPHA_DIRS[n_alpha]
    
    # Find the overlaps directory
    overlaps_dir = find_overlaps_dir(alpha_dir, N, beta)

    if overlaps_dir is None:
        print(f"Could not find overlaps directory for {alpha_dir}")
        print("Skipping single run analysis...")
    else:
        r_target = 16

        # Try new format first: overlaps_r16.csv
        ofile_pattern = f"overlaps_r{r_target}.csv"
        ofiles = list(overlaps_dir.glob(ofile_pattern))
        
        # Fallback: old format
        if not ofiles:
            ofile_pattern = f"overlaps_N{N}_beta*_r{r_target}.csv"
            ofiles = list(overlaps_dir.glob(ofile_pattern))
        
        if not ofiles:
            print(f"File not found: overlaps_r{r_target}.csv")
            print("Skipping single run analysis...")
        else:
            ofile = ofiles[0]
            sweeps, overlaps = load_overlaps(ofile)
            sweeps_plot = fix_sweeps(sweeps)

            # FIGURE 3 : All mu components for a single run
            plt.figure(figsize=(10, 6))
            
            # Plot all components with thin colored lines
            for mu in range(overlaps.shape[1]):
                plt.plot(sweeps_plot, overlaps[:, mu], linewidth=0.8, alpha=0.5)
            
            # Find and highlight mu* (pattern with maximum final overlap)
            mu_star = np.argmax(np.abs(overlaps[-1]))
            plt.plot(sweeps_plot, overlaps[:, mu_star], linewidth=2,
                    label=f"$m^{{\\mu^*}}$ ($\\mu^*={mu_star}$)")
            
            plt.xlabel("Number of sweeps", fontsize=12)
            plt.ylabel("$m^\\mu$", fontsize=12)
            plt.title(f"All overlap components - N={N_label}, $\\beta={beta_label}$, r={r_target}", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best", fontsize=10)

            fig3_path = output_dir / f"all_mu_single_run_N{N_label}_beta{beta_label}.png"
            plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"FIGURE 3 (all_mu_single_run) saved: {fig3_path}")

            # ============================================================
            # FIGURE 4 : Heatmap of overlaps for a single run
            # ============================================================

            plt.figure(figsize=(10, 6))
            im = plt.imshow(overlaps.T, aspect="auto", origin="lower", cmap="viridis")
            plt.colorbar(im, label="$m^\\mu$")
            plt.xlabel("Sweeps", fontsize=12)
            plt.ylabel("Pattern index $\\mu$", fontsize=12)
            plt.title(f"Heatmap of overlaps - N={N_label}, $\\beta={beta_label}$, r={r_target}", fontsize=14)

            fig4_path = output_dir / f"heatmap_N{N_label}_beta{beta_label}.png"
            plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"FIGURE 4 (heatmap) saved: {fig4_path}")

            # ============================================================
            # FIGURE 5 : mu* (maximum overlap component) for all runs
            # ============================================================

            ofiles = get_overlap_files(overlaps_dir, N, beta)

            plt.figure(figsize=(10, 6))

            for ofile in ofiles:
                sweeps, ov = load_overlaps(ofile)
                mu_star = np.argmax(np.abs(ov[-1]))

                sweeps_plot = fix_sweeps(sweeps)
                plt.plot(sweeps_plot, ov[:, mu_star], alpha=0.4, linewidth=1)

            plt.xlabel("Number of sweeps", fontsize=12)
            plt.ylabel(r"$m^{\mu^*}$", fontsize=12)
            plt.title(f"Evolution of $\\mu^*$ for all runs - N={N_label}, $\\beta={beta_label}$", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xscale('log')

            fig5_path = output_dir / f"mu_star_all_runs_N{N_label}_beta{beta_label}.png"
            plt.savefig(fig5_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"FIGURE 5 (mu_star_all_runs) saved: {fig5_path}")

    print("\n" + "="*60)
    print("All figures have been successfully generated!")
    print(f"Output directory: {output_dir.resolve()}")
    print("\nList of figures generated:")
    print("  FIGURE 1: mean_overlap - Average overlap for each alpha")
    print("  FIGURE 2: phase_diagram - Final overlap vs alpha")
    print("  FIGURE 3: all_mu_single_run - All overlap components with mu* highlighted")
    print("  FIGURE 4: heatmap - Heatmap of overlaps")
    print("  FIGURE 5: mu_star_all_runs - Evolution of mu* for all runs")
    print("="*60)