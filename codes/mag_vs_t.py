import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
import glob
import os
import re
from collections import defaultdict

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

# ── Paramètres ────────────────────────────────────────
csv_folder = "../results/magnetizations"
cache_file = "../results/m2_cache.csv"
T_target   = 2.2
T_tol      = 0.01
CHUNK      = 100_000
log2_min   = 3

# ── Lecture ou cache ───────────────────────────────────
if os.path.exists(cache_file):
    print(f"Cache trouvé : chargement de {cache_file}")
    cache   = pd.read_csv(cache_file)
    results = {}
    for L, grp in cache.groupby("L"):
        results[int(L)] = (
            grp["N_sweeps"].values,
            grp["m2_mean"].values,
            grp["m2_err"].values,
        )
    L_vals = sorted(results.keys())
    print(f"✓ {len(L_vals)} valeurs de L chargées depuis le cache")
else:

    # ── Lecture des fichiers CSV ───────────────────────
    files = sorted(glob.glob(os.path.join(csv_folder, "*.csv")))
    if not files:
        raise ValueError(f"Aucun fichier trouvé dans {csv_folder}")

    accum_sum   = defaultdict(float)
    accum_sum2  = defaultdict(float)
    accum_count = defaultdict(int)

    for i, f in enumerate(files):
        # Extraire L du nom de fichier
        match = re.search(r"L(\d+)_r(\d+)", os.path.basename(f))
        if not match:
            continue
        L_file = int(match.group(1))

        print(f"Lecture fichier {i+1}/{len(files)} : {os.path.basename(f)}", end="\r")
        for chunk in pd.read_csv(f, chunksize=CHUNK, header=0,
                                 dtype={"T": np.float32,
                                        "N": np.int32,
                                        "m": np.float32}):
            chunk.columns = ["T", "N_sweeps", "m"]
            chunk = chunk[(np.abs(chunk["T"] - T_target) < T_tol) & (chunk["N_sweeps"] > 0)]
            if chunk.empty:
                continue
            chunk["m2"] = chunk["m"] ** 2
            grp = chunk.groupby("N_sweeps", sort=False)["m2"].agg(
                ["sum", lambda x: (x**2).sum(), "count"]
            )
            grp.columns = ["s1", "s2", "n"]
            for N, row in grp.iterrows():
                key = (L_file, int(N))
                accum_sum[key]   += row["s1"]
                accum_sum2[key]  += row["s2"]
                accum_count[key] += row["n"]

    print(f"\n✓ {len(files)} fichiers lus, {len(accum_sum)} combinaisons trouvées")
    if not accum_sum:
        raise ValueError(f"Aucune donnée pour T ≈ {T_target}")

    # ── Reconstruction avec moyenne sur [t/2, t] pour tous les N ──
    L_vals  = sorted({k[0] for k in accum_sum})
    results = {}
    rows    = []

    for L in L_vals:
        all_N_L   = sorted({k[1] for k in accum_sum if k[0] == L})
        all_N_arr = np.array(all_N_L)

        sums   = np.array([accum_sum  [(L, N)] for N in all_N_L])
        sums2  = np.array([accum_sum2 [(L, N)] for N in all_N_L])
        counts = np.array([accum_count[(L, N)] for N in all_N_L])

        pad = np.zeros(1)
        cs1 = np.concatenate([pad, np.cumsum(sums)])
        cs2 = np.concatenate([pad, np.cumsum(sums2)])
        cc  = np.concatenate([pad, np.cumsum(counts)])

        # Vectorisé sur tous les N
        i_lo = np.searchsorted(all_N_arr, all_N_arr // 2)
        i_hi = np.arange(len(all_N_arr))

        s1 = cs1[i_hi + 1] - np.where(i_lo > 0, cs1[i_lo], 0)
        s2 = cs2[i_hi + 1] - np.where(i_lo > 0, cs2[i_lo], 0)
        n  = cc [i_hi + 1] - np.where(i_lo > 0, cc [i_lo], 0)

        valid  = n >= 2
        N_arr  = all_N_arr[valid]
        s1, s2, n = s1[valid], s2[valid], n[valid]
        mean   = s1 / n
        var    = np.maximum((s2 / n - mean**2) * n / (n - 1), 0)
        m2_err = np.sqrt(var / n)

        results[L] = (N_arr, mean, m2_err)
        for N, m, e in zip(N_arr, mean, m2_err):
            rows.append({"L": L, "N_sweeps": int(N), "m2_mean": m, "m2_err": e})

        print(f"L={L:3d}  ✓ {len(N_arr)} points calculés")

    # ── Sauvegarde du cache ────────────────────────────
    pd.DataFrame(rows).to_csv(cache_file, index=False)
    print(f"✓ Cache sauvegardé : {cache_file}")

# ── Valeur théorique (Onsager) ─────────────────────────
arg    = 1 - np.sinh(2 / T_target) ** (-4)
mag_th = arg ** (1 / 8) if arg > 0 else 0.0

# ── Style seaborn ──────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
palette = sns.color_palette("viridis", len(L_vals))

# ── Figure ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

# Puissances de 2 jusqu'au-delà du max des données
N_max_data    = max(N_arr[-1] for _, (N_arr, _, _) in results.items())
log2_max_plot = int(np.ceil(np.log2(N_max_data)))
pow2_targets  = np.array([2**k for k in range(log2_min, log2_max_plot + 1)])

for (L, (N_arr, m2_mean, m2_err)), color in zip(sorted(results.items()), palette):
    idx = np.unique([np.argmin(np.abs(N_arr - t)) for t in pow2_targets
                     if t <= N_arr[-1]])
    if len(N_arr) - 1 not in idx:
        idx = np.sort(np.append(idx, len(N_arr) - 1))

    ax.errorbar(N_arr[idx], m2_mean[idx], yerr=m2_err[idx],
                fmt="o-", color=color, linewidth=1.8,
                elinewidth=1.0, capsize=3, capthick=1.0,
                markersize=5, label=f"$L = {L}$", zorder=3)

ax.axhline(mag_th**2, color=sns.color_palette("deep")[3],
           linestyle="--", linewidth=1.5,
           label=f"Onsager : $\\langle m^2 \\rangle = {mag_th**2:.3f}$",
           zorder=2)

# ── Axes ──────────────────────────────────────────────
ax.set_xscale("log", base=2)
ax.set_xlabel(r"$\log_2(N_\mathrm{sweeps})$", labelpad=8)
ax.set_ylabel(r"$\langle m^2 \rangle$", labelpad=8)
ax.set_title(
    f"$\\langle m^2 \\rangle$ vs. $\\log_2(N_\\mathrm{{sweeps}})$  —  $T/J = {T_target}$",
    pad=12)

ax.set_xticks(pow2_targets)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"${int(np.round(np.log2(x)))}$" if x > 0 else ""
))
ax.tick_params(axis='x', rotation=0)
ax.set_xticks([], minor=True)
ax.set_xlim(left=2**log2_min)

ax.legend(loc="best", framealpha=0.9, borderpad=0.7, labelspacing=0.4)
sns.despine(ax=ax, left=False, bottom=False)

plt.tight_layout()
plt.savefig("../results/m2_vs_logt.png", dpi=200, bbox_inches="tight")
plt.show()