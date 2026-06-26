import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import glob
import os
import re

# =============================================================================
# magnetization_vs_T.py
#
# Plots sqrt(<m²>) as a function of temperature T/J for the 2D Ising model.
#
# Input:  ../results/Ising/magnetization_matrix/T_{T}.csv
#         columns: sweep, r0, r1, ..., r63
# Output: ../results/Ising/magnetizations.svg
# =============================================================================

# ── Parameters ────────────────────────────────────────
csv_folder = "../results/Ising/magnetization_matrix"

# ── Load data ─────────────────────────────────────────
files = sorted(glob.glob(os.path.join(csv_folder, "T_*.csv")))
if not files:
    raise ValueError(f"No files found in {csv_folder}")

def load_last_mag(fpath):
    """Retourne le vecteur des |m| au dernier sweep (une valeur par réalisation)."""
    df     = pd.read_csv(fpath)
    r_cols = [c for c in df.columns if c.startswith('r')]
    last   = df.iloc[-1]
    return last[r_cols].to_numpy(dtype=float)

T_vals    = []
mags_by_T = {}

for f in files:
    m = re.search(r"T_([0-9.]+)\.csv", os.path.basename(f))
    if not m:
        continue
    T    = float(m.group(1))
    mags = load_last_mag(f)
    T_vals.append(T)
    mags_by_T[T] = mags

T_vals = np.array(sorted(T_vals))
n_real = len(mags_by_T[T_vals[0]])

print(f"Loaded {len(T_vals)} temperatures, {n_real} realizations each.")

# ── Compute mean and error ─────────────────────────────
rms_m     = np.zeros(len(T_vals))
err_rms_m = np.zeros(len(T_vals))
mags_all  = []

for i, T in enumerate(T_vals):
    mags     = mags_by_T[T]
    mags_all.append(mags)
    rms      = np.sqrt(np.mean(mags ** 2))
    std_m2   = np.std(mags ** 2, ddof=1)
    err      = std_m2 / (2 * rms * np.sqrt(n_real)) if rms > 0 else 0.0
    rms_m[i]     = rms
    err_rms_m[i] = err

# ── Theoretical curve (Onsager exact solution) ────────
Tc     = 2.269185
T_th   = np.linspace(0.001, 5.05, 2000)
arg    = 1 - np.sinh(2 / T_th) ** (-4)
mag_th = np.where(arg > 0, arg ** (1 / 8), 0.0)

# ── Plot ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
fig, ax = plt.subplots(figsize=(9, 5.5))

# Individual realizations — trajectories
for r in range(n_real):
    ax.plot(T_vals,
            [np.abs(mags_all[i][r]) for i in range(len(T_vals))],
            color="blue", alpha=0.15, linewidth=0.7,
            label=f"{n_real} realizations" if r == 0 else None,
            zorder=2)
    
# Mean with error bars
ax.errorbar(T_vals, rms_m, yerr=err_rms_m,
            fmt="o", color="black", linewidth=0,
            elinewidth=1.0, capsize=3, capthick=1.0,
            markersize=4, markeredgewidth=0,
            label="average", zorder=4)

# Critical temperature
ax.axvline(Tc, color="darkred",
           linestyle="--", linewidth=1.5, alpha=0.85,
           label=r"$T_c/J \approx 2.269$", zorder=5)

# Onsager exact solution
ax.plot(T_th, mag_th,
        color="darkgreen", linewidth=2.0,
        zorder=6, label="Onsager exact solution")

# ── Axes ──────────────────────────────────────────────
ax.set_xlim([0, max(T_vals) * 1.05])
ax.set_ylim([-0.02, 1.05])
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.tick_params(which="minor", length=3)
ax.tick_params(which="major", length=5)
ax.set_xlabel(r"$T\,/\,J$", labelpad=8)
ax.set_ylabel(r"$\sqrt{\langle m^2 \rangle}$", labelpad=8)
ax.set_title("Magnetization vs. temperature  —  L=100  —  2D Ising model", pad=12)
ax.legend(loc="upper right", framealpha=0.9, borderpad=0.7, labelspacing=0.4)
sns.despine(ax=ax, left=False, bottom=False)

plt.tight_layout()
plt.savefig("../results/Ising/magnetizations.svg", format='svg', dpi=200, bbox_inches="tight")
plt.show()



# ── Plot 2 : m (signed) — two branches ───────────────
fig2, ax2 = plt.subplots(figsize=(9, 5.5))

for r in range(n_real):
    ax2.plot(T_vals,
             [mags_all[i][r] for i in range(len(T_vals))],
             color="blue", alpha=0.15, linewidth=0.7,
             label=f"{n_real} realizations" if r == 0 else None,
             zorder=2)

mean_m   = np.array([np.mean(mags_all[i]) for i in range(len(T_vals))])
err_mean = np.array([np.std(mags_all[i], ddof=1) / np.sqrt(n_real) for i in range(len(T_vals))])

ax2.errorbar(T_vals, mean_m, yerr=err_mean,
             fmt="o", color="black", linewidth=0,
             elinewidth=1.0, capsize=3, capthick=1.0,
             markersize=4, markeredgewidth=0,
             label="average", zorder=4)

ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
ax2.axvline(Tc, color="darkred", linestyle="--", linewidth=1.5, alpha=0.85,
            label=r"$T_c/J \approx 2.269$", zorder=5)
ax2.plot(T_th,  mag_th, color="darkgreen", linewidth=2.0, zorder=6, label="Onsager (+)")
ax2.plot(T_th, -mag_th, color="darkgreen", linewidth=2.0, zorder=6, linestyle="--")

ax2.set_xlim([0, max(T_vals) * 1.05])
ax2.set_ylim([-1.05, 1.05])
ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax2.tick_params(which="minor", length=3)
ax2.tick_params(which="major", length=5)
ax2.set_xlabel(r"$T\,/\,J$", labelpad=8)
ax2.set_ylabel(r"$m$", labelpad=8)
ax2.set_title("Signed magnetization vs. temperature  —  L=100  —  2D Ising model", pad=12)
ax2.legend(loc="upper right", framealpha=0.9, borderpad=0.7, labelspacing=0.4)
sns.despine(ax=ax2, left=False, bottom=False)

plt.tight_layout()
plt.savefig("../results/Ising/magnetizations_signed.svg", format='svg', dpi=200, bbox_inches="tight")
plt.show()