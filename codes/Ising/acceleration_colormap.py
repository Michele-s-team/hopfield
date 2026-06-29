import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Read CSV
# ==========================================================
df = pd.read_csv("../results/Ising/speedup.csv")

# ==========================================================
# Pivot for heatmap
# ==========================================================
Z = df.pivot(index="N", columns="T", values="ratio")
Z = Z.sort_index().sort_index(axis=1)

# ==========================================================
# Figure with 2 panels
# ==========================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# ==========================================================
# (1) Heatmap
# ==========================================================
im = ax1.imshow(
    Z.values,
    origin="lower",
    aspect="auto",
    cmap="Reds",
    interpolation="nearest"
)

ax1.set_xlabel(r"$T/J$")
ax1.set_ylabel(r"$N = L^2$")

ax1.set_xticks(range(len(Z.columns)))
ax1.set_xticklabels([f"{t:.1f}" for t in Z.columns])

ax1.set_yticks(range(len(Z.index)))
ax1.set_yticklabels([str(n) for n in Z.index])

cbar = plt.colorbar(im, ax=ax1)
cbar.set_label("Speed-up factor")

# ==========================================================
# (2) Curves: speed-up vs T for each N
# ==========================================================
for N, group in df.groupby("N"):
    group = group.sort_values("T")
    ax2.plot(
        group["T"],
        group["ratio"],
        marker="o",
        linewidth=1.5,
        label=f"N={N}"
    )

ax2.set_xlabel(r"$T/J$")
ax2.set_ylabel("Speed-up factor")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()