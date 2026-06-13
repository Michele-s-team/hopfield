#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


# ============================================================
# ROOT
# ============================================================

root_dir = Path("../results")


# ============================================================
# UTIL: PARSING
# ============================================================

def extract_r(name):
    m = re.search(r"_r(\d+)\.csv$", name)
    return int(m.group(1)) if m else None


def extract_N(name):
    m = re.search(r"N(\d+)", name)
    return int(m.group(1)) if m else None


# ============================================================
# OVERLAP (bit-safe, consistent with decoding logic)
# ============================================================

def overlap_binary(config_row, pattern_row, N):

    n_full = N // 64
    remainder = N % 64

    corr = 0

    for b in range(n_full):
        s = int(str(config_row[f"block{b}"]))
        p = int(str(pattern_row[f"block{b}"]))

        xor = s ^ p
        corr += 64 - 2 * xor.bit_count()

    if remainder:
        s = int(str(config_row[f"block{n_full}"]))
        p = int(str(pattern_row[f"block{n_full}"]))

        mask = (1 << remainder) - 1
        xor = (s ^ p) & mask

        corr += remainder - 2 * xor.bit_count()

    return corr / N


# ============================================================
# MAX OVERLAP FOR FINAL CONFIG
# ============================================================

def compute_max_overlap(config_file, pattern_file, N):

    config_df = pd.read_csv(config_file, dtype=str)
    pattern_df = pd.read_csv(pattern_file, dtype=str)

    final_row = config_df.iloc[-1]

    m_max = -1.0

    for _, pattern_row in pattern_df.iterrows():

        m = abs(overlap_binary(final_row, pattern_row, N))
        if m > m_max:
            m_max = m

    return m_max


# ============================================================
# INDEX BUILDING
# ============================================================

def build_index(root_dir):

    index = {}

    for alpha_dir in root_dir.glob("alpha_*"):

        spins_dir = alpha_dir / "spins"
        patterns_dir = alpha_dir / "patterns"

        if not spins_dir.exists() or not patterns_dir.exists():
            continue

        config_files = sorted(spins_dir.glob("*.csv"))
        pattern_files = list(patterns_dir.glob("*.csv"))

        pattern_map = {}

        for p in pattern_files:
            r = extract_r(p.name)
            if r is not None:
                pattern_map[r] = p

        index[alpha_dir] = {
            "configs": config_files,
            "patterns": pattern_map
        }

    return index


# ============================================================
# DETECT N
# ============================================================

any_pattern = next(root_dir.rglob("patterns_N*.csv"))
N = extract_N(any_pattern.name)

print("[INFO] Detected N =", N)


# ============================================================
# INDEX
# ============================================================

index = build_index(root_dir)


# ============================================================
# LIMIT NUMBER OF ALPHA FOLDERS (NEW)
# ============================================================

N_ALPHA = 12  # <-- choose how many alpha folders to keep

sorted_items = sorted(index.items(), key=lambda x: float(x[0].name.split("_")[1]))
sorted_items = sorted_items[:N_ALPHA]


# ============================================================
# MAIN LOOP
# ============================================================

alpha_values = []
mean_overlaps = []

for alpha_dir, data in sorted_items:

    print("\n[PROCESS]", alpha_dir)

    configs = data["configs"]
    patterns = data["patterns"]

    overlaps = []

    for cfile in configs:

        r = extract_r(cfile.name)
        if r is None:
            continue

        if r not in patterns:
            continue

        pfile = patterns[r]

        m = compute_max_overlap(cfile, pfile, N)
        overlaps.append(m)

    if not overlaps:
        print("[WARN] no overlaps for", alpha_dir)
        continue

    alpha = float(alpha_dir.name.split("_")[1])

    alpha_values.append(alpha)
    mean_overlaps.append(np.mean(overlaps))

    print(f"alpha={alpha:.3f} mean={mean_overlaps[-1]:.4f}")


# ============================================================
# OUTPUT
# ============================================================

alpha_values = np.array(alpha_values)
mean_overlaps = np.array(mean_overlaps)

order = np.argsort(alpha_values)
alpha_values = alpha_values[order]
mean_overlaps = mean_overlaps[order]


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(alpha_values, mean_overlaps, "o-")

plt.xlim([0, 0.2])
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\langle \max_\mu m^\mu \rangle$")

plt.grid(True)
plt.tight_layout()
plt.show()