import numpy as np
import matplotlib.pyplot as plt

# ── Lecture du fichier ──────────────────────────────
data = np.loadtxt("../results/magnetizations_bits.csv", delimiter=",")

BJ   = data[:, 0]
mags = data[:, 1:]

T = 1.0 / BJ

# ── Tri par T croissant ─────────────────────────────
idx  = np.argsort(T)
T    = T[idx]
mags = mags[idx]

# ── Valeur absolue et moyenne ───────────────────────
abs_mags = np.abs(mags)
mean_abs_m = np.mean(abs_mags, axis=1)

# ── Tracé ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

n_real = abs_mags.shape[1]

# Nuage des |m| individuels
for r in range(n_real):
    ax.scatter(T, mags[:, r], color="steelblue", alpha=0.3, s=5)

# Courbe moyenne en noir
ax.plot(T, mean_abs_m, color="black", linewidth=2,
        label=r"$\langle |m| \rangle$")

# Température critique
ax.axvline(2.269185, color='r', linestyle='--', label=r"$T_c/J$")

plt.ylim([-1.05, 1.05])
plt.xlim([-0.05, 5.05])

ax.set_xlabel("T/J")
ax.set_ylabel(r"$|m|$")
ax.set_title("Absolute magnetization as a function of temperature")

plt.legend()
plt.tight_layout()
plt.savefig("../results/magnetizations.png", dpi=150)
plt.show()