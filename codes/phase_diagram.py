import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import glob
import os
import re


# =============================================================================
# magnetization_vs_T.py
#
# Plots the magnetization sqrt(<m²>) as a function of temperature T/J
# for the 2D Ising model simulated with a bitwise Metropolis algorithm.
#
# For each lattice size L and each temperature T:
#   - loads the magnetization at the last sweep from CSV files (one per (L, r))
#   - averages over all realizations r
#   - computes the standard error on sqrt(<m²>)
#
# Also displays:
#   - individual realization scatter points
#   - the Onsager exact solution for the infinite system
#   - the critical temperature Tc ≈ 2.269 J/k_B
#
# Input:  ../results/magnetizations/L{L}_r{r}.csv  (columns: T, N, m)
# Output: ../results/magnetizations.png
# =============================================================================

# ── Parameters ────────────────────────────────────────
csv_folder = "../results/magnetizations"

# ── Load data ─────────────────────────────────────────
# One file per (L, realization): extract L from filename
files = sorted(glob.glob(os.path.join(csv_folder, "L*_r*.csv")))
if not files:
    raise ValueError(f"No files found in {csv_folder}")

# Group files by L
from collections import defaultdict
files_by_L = defaultdict(list)
for f in files:
    match = re.search(r"L(\d+)_r(\d+)", os.path.basename(f))
    if match:
        files_by_L[int(match.group(1))].append(f)

def load_last_sweep(fpath):
    """For each temperature, keep only the measurement at the last sweep."""
    data = np.loadtxt(fpath, delimiter=',', skiprows=1)  # columns: T, N, m
    T_vals = np.unique(data[:, 0])
    rows = []
    for T in T_vals:
        mask  = data[:, 0] == T
        block = data[mask]
        rows.append(block[np.argmax(block[:, 1])])  # row with max N
    return np.array(rows)  # shape (n_T, 3): T, N, m

# ── Theoretical curve (Onsager exact solution) ────────
Tc   = 2.269185
T_th = np.linspace(0.001, 5.05, 2000)
arg  = 1 - np.sinh(2 / T_th) ** (-4)
mag_th = np.where(arg > 0, arg ** (1 / 8), 0.0)

# ── Plot ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
L_vals  = sorted(files_by_L.keys())
palette = sns.color_palette("viridis", len(L_vals))

fig, ax = plt.subplots(figsize=(9, 5.5))

for L, color in zip(L_vals, palette):
    flist   = files_by_L[L]
    n_real  = len(flist)
    all_data = [load_last_sweep(f) for f in flist]

    # Align on temperatures common to all realizations
    T_sets   = [set(d[:, 0]) for d in all_data]
    T_common = sorted(T_sets[0].intersection(*T_sets[1:]))
    T_common = np.array(T_common)

    mags = np.zeros((len(T_common), n_real))
    for r, d in enumerate(all_data):
        T_map = {row[0]: row[2] for row in d}
        for i, T in enumerate(T_common):
            mags[i, r] = T_map[T]

    # Sort by temperature
    idx  = np.argsort(T_common)
    T    = T_common[idx]
    mags = mags[idx]

    # sqrt(<m²>) with propagated error
    rms_m     = np.sqrt(np.mean(mags ** 2, axis=1))
    std_m2    = np.std(mags ** 2, axis=1, ddof=1)
    err_rms_m = std_m2 / (2 * rms_m * np.sqrt(n_real))

    # Individual realizations (light scatter)
    for r in range(n_real):
        ax.scatter(T, np.abs(mags[:, r]),
                   color=color, alpha=0.2, s=8, linewidths=0,
                   label=f"$L={L}$ realizations" if r == 0 else None,
                   zorder=2)

    # Mean with error bars
    ax.errorbar(T, rms_m, yerr=err_rms_m,
                fmt="o-", color=color, linewidth=1.6,
                elinewidth=1.0, capsize=3, capthick=1.0,
                markersize=4, markeredgewidth=0,
                label=f"$L = {L}$", zorder=4)

# Critical temperature
ax.axvline(Tc, color=sns.color_palette("deep")[3],
           linestyle="--", linewidth=1.3, alpha=0.85,
           label=r"$T_c/J \approx 2.269$", zorder=5)

# Onsager exact solution
ax.plot(T_th, mag_th,
        color=sns.color_palette("deep")[2], linewidth=2.0,
        zorder=6, label="Onsager exact solution")

# ── Axes ──────────────────────────────────────────────
ax.set_xlim([0, 4.05])
ax.set_ylim([-0.002, 1.05])
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.tick_params(which="minor", length=3)
ax.tick_params(which="major", length=5)
ax.set_xlabel(r"$T\,/\,J$", labelpad=8)
ax.set_ylabel(r"$\sqrt{\langle m^2 \rangle}$", labelpad=8)
ax.set_title("Magnetization vs. temperature  —  2D Ising model", pad=12)
ax.legend(loc="upper right", framealpha=0.9, borderpad=0.7, labelspacing=0.4)
sns.despine(ax=ax, left=False, bottom=False)

plt.tight_layout()
plt.savefig("../results/magnetizations.png", dpi=200, bbox_inches="tight")
plt.show()