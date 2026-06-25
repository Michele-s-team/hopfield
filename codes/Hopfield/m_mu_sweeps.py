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


def fix_sweeps(sweeps):
    sweeps_plot = sweeps.copy().astype(float)
    if sweeps_plot[0] == 0 and len(sweeps_plot) > 1:
        sweeps_plot[0] = sweeps_plot[1] / 2.0  # moitié du gap suivant
    return sweeps_plot


# ============================================================
# LOAD OVERLAPS FROM PRECOMPUTED CSV
# ============================================================

def load_overlaps(overlap_file):
    """
    Lit un fichier overlaps_r{r}.csv.
    Retourne (sweeps: 1D int array, overlaps: 2D float array (n_sweeps, n_patterns)).
    """
    df = pd.read_csv(overlap_file)
    sweeps = df["sweep"].to_numpy(dtype=int)
    m_cols = [c for c in df.columns if c.startswith("m_")]
    overlaps = df[m_cols].to_numpy(dtype=float)
    return sweeps, overlaps


# ============================================================
# PROCESS ONE ALPHA  (depuis overlaps précalculés)
# ============================================================

def process_alpha(alpha_dir):
    """
    Pour chaque réalisation r disponible dans alpha_dir/overlaps/,
    calcule max_mu |m^mu| à chaque sweep.
    Retourne (sweeps, mean_curve, std_curve).
    """
    overlaps_dir = alpha_dir / "overlaps"
    if not overlaps_dir.exists():
        print(f"  [WARN] pas de dossier overlaps dans {alpha_dir.name}")
        return None, None, None

    overlap_files = sorted(overlaps_dir.glob("overlaps_r*.csv"))
    if not overlap_files:
        return None, None, None

    curves = []
    sweeps = None

    for j, ofile in enumerate(overlap_files, start=1):
        print(f"    realization {j}/{len(overlap_files)}", end="\r", flush=True)

        sw, ov = load_overlaps(ofile)
        curve = np.max(np.abs(ov), axis=1)   # max_mu |m^mu| pour chaque sweep
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

    # ============================================================
    # CHOIX DES DOSSIERS ALPHA
    # ============================================================

    ALL_ALPHA_DIRS = sorted(root_dir.glob("alpha_*"))

    # exemples :
    # ALPHA_SELECTION = [1]
    # ALPHA_SELECTION = [2, 3]
    # ALPHA_SELECTION = [1, 4, 7]

    ALPHA_SELECTION = [1, 2, 3, 4, 5]

    alpha_dirs = [
        ALL_ALPHA_DIRS[i - 1]
        for i in ALPHA_SELECTION
        if 1 <= i <= len(ALL_ALPHA_DIRS)
    ]

    print("\nSelected alpha folders:")
    for d in alpha_dirs:
        print(" ", d.name)

    # ----------------------------------------------------------
    # Plot 1 : max_mu |m^mu| moyenné sur les réalisations
    # ----------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sweeps_plot_last = None

    for i, alpha_dir in enumerate(alpha_dirs, start=1):

        alpha = float(alpha_dir.name.split("_")[1])
        print(f"[{i}/{len(alpha_dirs)}] processing {alpha_dir.name}")

        sweeps, mean_curve, std_curve = process_alpha(alpha_dir)

        if sweeps is None:
            continue

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

    # ----------------------------------------------------------
    # Plots 2 & 3 : zoom sur un alpha particulier
    # ----------------------------------------------------------

    n_alpha = 3   # index dans ALL_ALPHA_DIRS (0-based)
    alpha_dir = ALL_ALPHA_DIRS[n_alpha]

    if not alpha_dirs:
        raise RuntimeError("No alpha folder selected.")

    alpha = float(alpha_dir.name.split("_")[1])
    print(f"\nalpha = {alpha:.3f}")

    overlaps_dir = alpha_dir / "overlaps"

    # ----------------------------------------------------------
    # Plot 2 : tous les m^mu pour une réalisation donnée
    # ----------------------------------------------------------

    r_target = 41

    ofile = overlaps_dir / f"overlaps_r{r_target}.csv"
    sweeps, overlaps = load_overlaps(ofile)

    sweeps_plot = fix_sweeps(sweeps)
    n_patterns  = overlaps.shape[1]
    dominant    = np.argmax(np.abs(overlaps[-1, :]))

    # infer N from pattern filename (pour le titre)
    pattern_files = list((alpha_dir / "patterns").glob(f"*_r{r_target}.csv"))
    N_label = re.search(r"N(\d+)", pattern_files[0].name).group(1) if pattern_files else "?"

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

    ofiles   = sorted(overlaps_dir.glob("overlaps_r*.csv"))
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