import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import mplcursors
import glob
import os

def plot_magnetization(csv_folder: str, base_name: str, T: float, N: int = None):
    # Trouve tous les fichiers de réalisations
    pattern = os.path.join(csv_folder, base_name + "_r*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise ValueError(f"Aucun fichier trouvé avec le pattern : {pattern}")

    n_real = len(files)
    colors = [hsv_to_rgb([i / n_real, 0.55, 0.75]) for i in range(n_real)]

    fig, ax = plt.subplots(figsize=(13, 5))
    lines = []

    for r, fpath in enumerate(files):
        # Chaque fichier : colonnes T, N, m
        data = np.loadtxt(fpath, delimiter=',', skiprows=1)

        # Filtre sur T
        mask = data[:, 0] == T

        # Filtre optionnel sur N
        if N is not None:
            mask &= data[:, 1] == N

        rows = data[mask]

        if rows.size == 0:
            print(f"[warning] T={T} introuvable dans {fpath}, ignoré.")
            continue

        sweeps = rows[:, 1].astype(int) if N is None else rows[:, 2]  # N ou m selon cas
        mags   = rows[:, 2] if N is None else rows[:, 2]

        line, = ax.plot(sweeps, mags, lw=0.8, color=colors[r])
        lines.append((line, r))

    if not lines:
        raise ValueError(f"Aucune donnée trouvée pour T={T}.")

    # Hover tooltip showing realization index
    cursor = mplcursors.cursor([l for l, _ in lines], hover=True)
    @cursor.connect("add")
    def on_add(sel):
        r = next(idx for l, idx in lines if l is sel.artist)
        sel.annotation.set_text(f"realization {r}")
        sel.annotation.get_bbox_patch().set(fc=colors[r], alpha=0.8)

    plt.xscale('log')
    plt.xlim([sweeps[1], sweeps[-1]])
    plt.ylim([-1.05, 1.05])
    ax.set_xlabel("Sweeps")
    ax.set_ylabel("Magnetization")
    ax.set_title(f"T = {T}  —  {n_real} realizations")
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    plt.tight_layout()
    plt.show()

plot_magnetization(
    csv_folder="../results/magnetizations",
    base_name="magnetizations_bits",
    T=1.5
)