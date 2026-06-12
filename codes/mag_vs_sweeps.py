import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
import glob
import os
import re
from collections import defaultdict

# =============================================================================
# m2_vs_logt.py
#
# Plots the time-averaged <m²> as a function of log2(N_sweeps)
# for the 2D Ising model at a fixed temperature T/J, for several lattice sizes L
#
# For each lattice size L and each number of sweeps t:
#   - averages <m²> over all realizations and over the time window [t/2, t]
#   - computes the standard error on <m²>
#   - displays only points closest to integer powers of 2 for readability
#
# Results are cached to avoid reprocessing large CSV files on subsequent runs.
# Delete the cache file to force recomputation.
#
# Input:  ../results/magnetizations/L{L}_r{r}.csv  (columns: T, N, m)
# Cache:  ../results/m2_cache.csv
# Output: ../results/m2_vs_logt.png
# =============================================================================


# ── Parameters ────────────────────────────────────────
csv_folder = "../results/magnetizations"
cache_file = "../results/m2_cache.csv"
T_target   = 2.2
T_tol      = 0.01
CHUNK      = 100_000
log2_min   = 3

# ── Load from cache or recompute ──────────────────────
if os.path.exists(cache_file):
    print(f"Cache found: loading {cache_file}")
    cache   = pd.read_csv(cache_file)
    results = {}
    for L, grp in cache.groupby("L"):
        results[int(L)] = (
            grp["N_sweeps"].values,
            grp["m2_mean"].values,
            grp["m2_err"].values,
        )
    L_vals = sorted(results.keys())
    print(f"✓ {len(L_vals)} L values loaded from cache")

else:
    # ── Read CSV files ─────────────────────────────────
    files = sorted(glob.glob(os.path.join(csv_folder, "*.csv")))
    if not files:
        raise ValueError(f"No files found in {csv_folder}")

    # Accumulators: store sum, sum of squares, and count per (L, N_sweeps)
    accum_sum   = defaultdict(float)
    accum_sum2  = defaultdict(float)
    accum_count = defaultdict(int)

    for i, f in enumerate(files):
        # Extract L from filename (e.g. L30_r0.csv)
        match = re.search(r"L(\d+)_r(\d+)", os.path.basename(f))
        if not match:
            continue
        L_file = int(match.group(1))

        print(f"Reading file {i+1}/{len(files)} : {os.path.basename(f)}", end="\r")
        for chunk in pd.read_csv(f, chunksize=CHUNK, header=0,
                                 dtype={"T": np.float32,
                                        "N": np.int32,
                                        "m": np.float32}):
            chunk.columns = ["T", "N_sweeps", "m"]

            # Filter on temperature and discard N=0 (initial state)
            chunk = chunk[(np.abs(chunk["T"] - T_target) < T_tol) & (chunk["N_sweeps"] > 0)]
            if chunk.empty:
                continue

            chunk["m2"] = chunk["m"] ** 2

            # Aggregate per N_sweeps in a single vectorized pass
            grp = chunk.groupby("N_sweeps", sort=False)["m2"].agg(
                ["sum", lambda x: (x**2).sum(), "count"]
            )
            grp.columns = ["s1", "s2", "n"]
            for N, row in grp.iterrows():
                key = (L_file, int(N))
                accum_sum[key]   += row["s1"]
                accum_sum2[key]  += row["s2"]
                accum_count[key] += row["n"]

    print(f"\n✓ {len(files)} files read, {len(accum_sum)} (L, N_sweeps) combinations found")
    if not accum_sum:
        raise ValueError(f"No data found for T = {T_target}")

    # ── Compute time-averaged <m²> over [t/2, t] for each N ──
    # This follows the method of averaging over the second half of sweeps,
    # which suppresses noise while remaining statistically honest.
    L_vals  = sorted({k[0] for k in accum_sum})
    results = {}
    rows    = []

    for L in L_vals:
        all_N_L   = sorted({k[1] for k in accum_sum if k[0] == L})
        all_N_arr = np.array(all_N_L)

        sums   = np.array([accum_sum  [(L, N)] for N in all_N_L])
        sums2  = np.array([accum_sum2 [(L, N)] for N in all_N_L])
        counts = np.array([accum_count[(L, N)] for N in all_N_L])

        # Build prefix sums for O(1) window queries
        pad = np.zeros(1)
        cs1 = np.concatenate([pad, np.cumsum(sums)])
        cs2 = np.concatenate([pad, np.cumsum(sums2)])
        cc  = np.concatenate([pad, np.cumsum(counts)])

        # For each t, window starts at first N >= t//2
        i_lo = np.searchsorted(all_N_arr, all_N_arr // 2)
        i_hi = np.arange(len(all_N_arr))

        s1 = cs1[i_hi + 1] - np.where(i_lo > 0, cs1[i_lo], 0)
        s2 = cs2[i_hi + 1] - np.where(i_lo > 0, cs2[i_lo], 0)
        n  = cc [i_hi + 1] - np.where(i_lo > 0, cc [i_lo], 0)

        # Keep only points with enough samples for a meaningful error estimate
        valid  = n >= 2
        N_arr  = all_N_arr[valid]
        s1, s2, n = s1[valid], s2[valid], n[valid]
        mean   = s1 / n
        var    = np.maximum((s2 / n - mean**2) * n / (n - 1), 0)  # corrected variance
        m2_err = np.sqrt(var / n)                                   # standard error

        results[L] = (N_arr, mean, m2_err)
        for N, m, e in zip(N_arr, mean, m2_err):
            rows.append({"L": L, "N_sweeps": int(N), "m2_mean": m, "m2_err": e})

        print(f"L={L:3d}  ✓ {len(N_arr)} points computed")

    # ── Save cache ────────────────────────────────────
    pd.DataFrame(rows).to_csv(cache_file, index=False)
    print(f"✓ Cache saved: {cache_file}")

# ── Theoretical value (Onsager exact solution) ────────
arg    = 1 - np.sinh(2 / T_target) ** (-4)
mag_th = arg ** (1 / 8) if arg > 0 else 0.0

# ── Plot ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
palette = sns.color_palette("viridis", len(L_vals))

fig, ax = plt.subplots(figsize=(9, 5.5))

# Powers of 2 up to and including the one just above the data maximum
N_max_data    = max(N_arr[-1] for _, (N_arr, _, _) in results.items())
log2_max_plot = int(np.ceil(np.log2(N_max_data)))
pow2_targets  = np.array([2**k for k in range(log2_min, log2_max_plot + 1)])

for (L, (N_arr, m2_mean, m2_err)), color in zip(sorted(results.items()), palette):
    # Select the point closest to each power of 2
    idx = np.unique([np.argmin(np.abs(N_arr - t)) for t in pow2_targets
                     if t <= N_arr[-1]])
    # Always include the last point even if it falls between two powers of 2
    if len(N_arr) - 1 not in idx:
        idx = np.sort(np.append(idx, len(N_arr) - 1))

    ax.errorbar(N_arr[idx], m2_mean[idx], yerr=m2_err[idx],
                fmt="o-", color=color, linewidth=1.8,
                elinewidth=1.0, capsize=3, capthick=1.0,
                markersize=5, label=f"$L = {L}$", zorder=3)

# Onsager theoretical value
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

# X ticks at exact powers of 2, labeled by their exponent
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