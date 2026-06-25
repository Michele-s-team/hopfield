#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ============================================================
# CONFIG
# ============================================================

root_dir = Path("../results")

# choisir quels "lots alpha" analyser (ex: lots 2 et 3)
SELECT_LOTS = {2, 3}

# limiter le nombre de dossiers alpha (optionnel)
N_ALPHA = None  # ex: 17 ou None pour tout garder


# ============================================================
# PARSING
# ============================================================

def get_r(name):
    m = re.search(r"_r(\d+)\.csv$", name)
    return int(m.group(1)) if m else None


def get_N(name):
    m = re.search(r"N(\d+)", name)
    return int(m.group(1)) if m else None


# ============================================================
# OVERLAP
# ============================================================

def overlap(config, pattern, N):
    n64 = N // 64
    rem = N % 64
    corr = 0

    for b in range(n64):
        s = int(config[f"block{b}"])
        p = int(pattern[f"block{b}"])
        xor = s ^ p
        corr += 64 - 2 * xor.bit_count()

    if rem:
        s = int(config[f"block{n64}"])
        p = int(pattern[f"block{n64}"])
        mask = (1 << rem) - 1
        xor = (s ^ p) & mask
        corr += rem - 2 * xor.bit_count()

    return corr / N


# ============================================================
# INDEX
# ============================================================

index = {}

for alpha_dir in root_dir.glob("alpha_*"):

    # extraction lot id si présent (ex: alpha_0.120_2)
    parts = alpha_dir.name.split("_")
    try:
        lot_id = int(parts[-1])
    except:
        continue

    if SELECT_LOTS and lot_id not in SELECT_LOTS:
        continue

    spins = sorted((alpha_dir / "spins").glob("*.csv"))
    patterns = {
        get_r(p.name): p
        for p in (alpha_dir / "patterns").glob("*.csv")
    }

    index[alpha_dir] = (spins, patterns)


# ============================================================
# DETECT N
# ============================================================

any_pattern = next(root_dir.rglob("patterns_N*.csv"))
N = get_N(any_pattern.name)

print("[INFO] N =", N)


# ============================================================
# LIMIT ALPHAS
# ============================================================

items = sorted(index.items(), key=lambda x: float(x[0].name.split("_")[1]))

if N_ALPHA:
    items = items[:N_ALPHA]


# ============================================================
# MAIN LOOP
# ============================================================

alphas = []
errors = []

for i, (alpha_dir, (configs, patterns)) in enumerate(items, 1):

    print(f"\n[{i}/{len(items)}] {alpha_dir.name}")

    vals = []

    for cfg_file in configs:

        r = get_r(cfg_file.name)
        if r not in patterns:
            continue

        cfg = pd.read_csv(cfg_file, dtype=str).iloc[-1]
        pat = pd.read_csv(patterns[r], dtype=str)

        best = max(
            abs(overlap(cfg, row, N))
            for _, row in pat.iterrows()
        )

        vals.append((1 - best) / 2)

    if not vals:
        print("  [WARN] empty")
        continue

    alpha = float(alpha_dir.name.split("_")[1])

    alphas.append(alpha)
    errors.append(np.mean(vals))

    print(f"  alpha={alpha:.3f} error={errors[-1]:.4f}")


# ============================================================
# SORT
# ============================================================

alphas = np.array(alphas)
errors = np.array(errors)

idx = np.argsort(alphas)
alphas = alphas[idx]
errors = errors[idx]


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(alphas, errors, "o-")

plt.xlim([0, 0.2])
plt.xlabel(r"$\alpha$")
plt.ylabel("error rate")
plt.grid()
plt.tight_layout()
plt.show()