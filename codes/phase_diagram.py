import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d

# ── Style ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#cccccc",
    "axes.labelcolor":   "#222222",
    "axes.titlecolor":   "#222222",
    "axes.grid":         True,
    "grid.color":        "#eeeeee",
    "grid.linewidth":    0.8,
    "xtick.color":       "#555555",
    "ytick.color":       "#555555",
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "axes.labelsize":    13,
    "axes.titlesize":    14,
    "legend.facecolor":  "white",
    "legend.edgecolor":  "#cccccc",
    "legend.fontsize":   10.5,
    "savefig.facecolor": "white",
})

# ── Données ───────────────────────────────────────────
data = np.loadtxt("../results/magnetizations_bits.csv", delimiter=",")
BJ   = data[:, 0]
mags = data[:, 1:]
T    = 1.0 / BJ

idx  = np.argsort(T)
T    = T[idx]
mags = mags[idx]

# ── sqrt(<m²>) ────────────────────────────────────────
rms_m     = np.sqrt(np.mean(mags**2, axis=1))
n_real    = mags.shape[1]
# erreur standard sur sqrt(<m²>) via propagation : std(m²) / (2*sqrt(<m²>) * sqrt(N))
m2        = mags**2
std_m2    = np.std(m2, axis=1, ddof=1)
err_rms_m = std_m2 / (2 * rms_m * np.sqrt(n_real))

# ── Courbe théorique Onsager ──────────────────────────
Tc   = 2.269185
T_th = np.linspace(0.001, 5.05, 2000)
arg = 1 - np.sinh(2 / T_th)**(-4)
mag_th = np.where(arg > 0, arg**(1/8), 0.0)

# ── Figure ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

# Réalisations individuelles
for r in range(n_real):
    ax.scatter(
        T, np.abs(mags[:, r]),
        color="#1a5fa3", alpha=0.3, s=10, linewidths=0,
        label=f"{n_real} realizations" if r == 0 else None,
        zorder=2,
    )

# Température critique
ax.axvline(Tc, color="#b41c1c", linestyle="--", linewidth=1.3, alpha=0.85,
           label=r"$T_c/J \approx 2.269$", zorder=4)
"""
# Courbe lissée
smoothed = gaussian_filter1d(rms_m, sigma=2)
ax.plot(T, smoothed, color="#000000", linewidth=1.8, alpha=0.4, zorder=5)
"""

# Solution exacte Onsager
ax.plot(
    T_th, mag_th,
    color="#05532a", linewidth=2.0, linestyle="-",
    zorder=7, label="Onsager exact solution",
)


# sqrt(<m²>) avec barres d'erreur
ax.errorbar(
    T, rms_m, yerr=err_rms_m,
    fmt="o", color="#000000",
    ecolor="#0000006b", elinewidth=1.1,
    capsize=3, capthick=1.1,
    markersize=4.5, markeredgewidth=0,
    zorder=6, label=r"$\sqrt{\langle m^2 \rangle}$",
)
print(T)
print(rms_m)

# ── Axes ──────────────────────────────────────────────
ax.set_xlim([0, 5.05])
ax.set_ylim([-0.002, 1.05])
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.tick_params(which="minor", length=3, color="#aaaaaa")
ax.tick_params(which="major", length=5)

ax.set_xlabel(r"$T\,/\,J$", labelpad=8)
ax.set_ylabel(r"$\sqrt{\langle m^2 \rangle}$", labelpad=8)
ax.set_title("Magnetization vs. temperature  —  2D Ising model", pad=12)

ax.legend(loc="upper right", framealpha=0.9, borderpad=0.7, labelspacing=0.4)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)

plt.tight_layout()
plt.savefig("../results/magnetizations.png", dpi=200, bbox_inches="tight")
plt.show()