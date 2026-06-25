import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import mplcursors
import glob
import os

def plot_magnetization(csv_folder: str, base_name: str, T: float, L: int, N: int = None):
    pattern = os.path.join(csv_folder, f"{base_name}_L{L}_r*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(f"Aucun fichier trouvé avec le pattern : {pattern}")

    n_real = len(files)
    colors = [hsv_to_rgb([i / n_real, 0.55, 0.75]) for i in range(n_real)]

    fig, ax = plt.subplots(figsize=(12, 4))
    lines = []

    for r, fpath in enumerate(files):
        data = np.loadtxt(fpath, delimiter=',', skiprows=1)
        if data.ndim == 1:
            data = data[np.newaxis, :]

        mask = data[:, 0] == T
        if N is not None:
            mask &= data[:, 1] == N

        rows = data[mask]
        if rows.size == 0:
            print(f"[warning] T={T} introuvable dans {fpath}, ignoré.")
            continue

        sweeps = rows[:, 1].astype(int)
        mags   = rows[:, 2]

        line, = ax.plot(sweeps, mags, lw=0.8, color=colors[r])
        lines.append((line, r))

    if not lines:
        raise ValueError(f"Aucune donnée trouvée pour T={T}, L={L}.")

    cursor = mplcursors.cursor([l for l, _ in lines], hover=True)

    @cursor.connect("add")
    def on_add(sel):
        r = next(idx for l, idx in lines if l is sel.artist)
        sel.annotation.set_text(f"realization {r}")
        sel.annotation.get_bbox_patch().set(fc=colors[r], alpha=0.8)

    plt.xscale('log')
    ax.set_xlim([sweeps[1], 2e5])
    ax.set_ylim([-1.05, 1.05])
    ax.set_xlabel("Sweeps", fontsize=13)
    ax.set_ylabel("$m$", fontsize=13)
    ax.set_title(f"T = {T},  L = {L}  —  {n_real} realizations", fontsize=14)
    ax.tick_params(labelsize=11, length=5, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # Sauvegarde dans le dossier parent du parent de csv_folder
    out_dir = os.path.abspath(os.path.join(csv_folder, '..', '..'))
    out_path = os.path.join(out_dir, f"evolution_magnetization_T{T}_L{L}.svg")
    plt.savefig(out_path, format='svg', dpi=200, bbox_inches='tight')
    print(f"Figure sauvegardée : {out_path}")

    plt.show()

plot_magnetization(
    csv_folder="../results/magnetizations/T=1.5",
    base_name="magnetizations_bits",
    T=1.5,
    L=100
)