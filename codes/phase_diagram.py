import numpy as np
import matplotlib.pyplot as plt

# ── Lecture du fichier ──────────────────────────────
data = np.loadtxt("../results/magnetizations.csv", delimiter=",")

BJ   = data[:, 0]
mags = data[:, 1:]

T = 1.0 / BJ

# ── Tri par T croissant ─────────────────────────────
idx  = np.argsort(T)
T    = T[idx]
mags = mags[idx]

# ── Tracé ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

n_real = mags.shape[1]
for r in range(n_real):
    ax.scatter(T, mags[:, r], color="steelblue", alpha=0.3, s=5)

plt.axvline(2.269185, color='r', label=r"$T_c/J$")
plt.ylim([-1.05,1.05])
plt.xlim([-0.05,4.05])

ax.set_xlabel("T/J")
ax.set_ylabel("m")
plt.legend()
ax.set_title("Magnetization a function of the temperature")
plt.tight_layout()
plt.savefig("../results/magnetizations.png", dpi=150)
plt.show()