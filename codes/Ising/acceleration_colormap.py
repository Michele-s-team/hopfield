import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Read CSV
# ==========================================================
df = pd.read_csv("../results/Ising/speedup.csv")

# ==========================================================
# Pivot table
# ==========================================================
Z = df.pivot(index="N", columns="T", values="ratio")

fig, ax = plt.subplots(figsize=(7, 4.5))

# Toutes les cases ont la même taille
im = ax.imshow(
    Z.values,
    origin="lower",
    aspect="equal",      # carrés
    cmap="Reds",
    interpolation="nearest"
)

# Axes
ax.set_xlabel(r"$T/J$")
ax.set_ylabel(r"$N=L^2$")

# Ticks
ax.set_xticks(range(len(Z.columns)))
ax.set_xticklabels([f"{t:.1f}" for t in Z.columns])

ax.set_yticks(range(len(Z.index)))
ax.set_yticklabels([str(n) for n in Z.index])

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Speed-up factor")

plt.tight_layout()
plt.show()