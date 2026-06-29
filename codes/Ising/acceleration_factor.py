import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


plt.rcParams.update({
    "text.usetex"                : True,
    "font.family"                : "serif",
    "font.serif"                 : ["Computer Modern Roman"],
    "axes.formatter.use_mathtext": True
    })

# ── Data ────────────────────────────────────────────────────────────────────
df = pd.read_csv("../results/Ising/speedup_save.csv")

# ── Couleurs distinctes par N ────────────────────────────────────────────────
N_vals  = sorted(df["N"].unique())
palette = plt.cm.plasma(np.linspace(0.1, 0.85, len(N_vals)))
color   = {N: palette[i] for i, N in enumerate(N_vals)}

markers = ["o", "s", "D", "^", "v", "P", "X"]
marker  = {N: markers[i % len(markers)] for i, N in enumerate(N_vals)}


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — Heatmap
# ════════════════════════════════════════════════════════════════════════════
Z = df.pivot(index="N", columns="T", values="ratio")
Z = Z.sort_index().sort_index(axis=1)

fig1, ax = plt.subplots(figsize=(4, 3.2))

im = ax.imshow(
    Z.values,
    origin="lower",
    aspect="auto",
    cmap="viridis",
    interpolation="nearest",
    vmin=np.min(Z.values),
    vmax=np.max(Z.values),
    rasterized=True,
)

# Axes ticks
ax.set_xticks(range(len(Z.columns)))
ax.set_xticklabels([f"{t:.1f}" for t in Z.columns], rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(Z.index)))
ax.set_yticklabels([str(n) for n in Z.index], fontsize=8)

ax.set_xlabel(r"$T\,/\,J$")
ax.set_ylabel(r"$N = L^2$")

cbar = fig1.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("Speed-up factor", fontsize=10)
cbar.ax.tick_params(labelsize=8)


fig1.savefig("../results/Ising/speedup_heatmap_ising.pdf", bbox_inches="tight", pad_inches=0.05)
fig1.tight_layout()
fig1.savefig("../results/Ising/speedup_heatmap_ising.png", bbox_inches="tight", dpi=300)
print("Saved: speedup_heatmap.pdf / .png")


# ── Style global ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "xtick.direction"            : "in",
    "ytick.direction"            : "in",
    "xtick.top"                  : True,
    "ytick.right"                : True,
    "xtick.major.size"           : 5,
    "ytick.major.size"           : 5,
    "xtick.minor.size"           : 3,
    "ytick.minor.size"           : 3,
    "xtick.major.width"          : 0.8,
    "ytick.major.width"          : 0.8,
    "xtick.minor.visible"        : True,
    "ytick.minor.visible"        : True,
    "axes.linewidth"             : 0.8,
    "axes.spines.top"            : True,
    "axes.spines.right"          : True,
    "legend.frameon"             : False,
    "legend.fontsize"            : 8,
    "legend.handlelength"        : 2.0,
    "legend.labelspacing"        : 0.3,
    "legend.handletextpad"       : 0.5,
    "font.size"                  : 10,
    "axes.labelsize"             : 11,
    "xtick.labelsize"            : 9,
    "ytick.labelsize"            : 9,
    "lines.linewidth"            : 1.2,
})



# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Courbes speed-up vs T
# ════════════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(4, 3.2))

for N in N_vals:
    group = df[df["N"] == N].sort_values("T")
    ax.plot(
        group["T"], group["ratio"],
        marker=marker[N],
        markersize=4,
        color=color[N],
        linewidth=1.2,
        label=f"$N={N}$",
    )

ax.set_xlabel(r"$T\,/\,J$")
ax.set_ylabel("Speed-up factor")
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(True, which="major", alpha=0.25, linewidth=0.5)
ax.legend(framealpha=0.9, edgecolor="0.7", handlelength=1.8, ncol=2)


fig2.savefig("../results/Ising/speedup_curves_ising.pdf", bbox_inches="tight", pad_inches=0.05)
fig2.tight_layout()
fig2.savefig("../results/Ising/speedup_curves_ising.png", bbox_inches="tight", dpi=300)
print("Saved: speedup_curves.pdf / .png")