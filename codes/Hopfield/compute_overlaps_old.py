#!/usr/bin/env python3
"""
For each selected alpha_* directory and each realisation r:
  - reads spins/spins_r{r}.csv
  - reads patterns/patterns_N{N}_r{r}.csv
  - computes the bit-safe overlap of every sweep with every pattern
  - saves to overlaps/overlaps_r{r}.csv

Output DataFrame columns: sweep | m_0 | m_1 | ... | m_{p-1}

Existing files are skipped (set SKIP_EXISTING = False to overwrite).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re

# ============================================================
# CONFIGURATION
# ============================================================

root_dir = Path("../save_old/25_June/")

print("ROOT DIR:", root_dir.resolve())
print("EXISTS:", root_dir.exists())
print("CONTENT:", list(root_dir.glob("*"))[:10])

ALPHA_SELECTION = [1, 2, 3, 4, 5, 6]
SKIP_EXISTING = True

# ============================================================
# HELPERS
# ============================================================

def extract_r(name: str):
    m = re.search(r"_r(\d+)\.csv$", name)
    return int(m.group(1)) if m else None

def extract_N(name: str):
    m = re.search(r"N(\d+)", name)
    return int(m.group(1)) if m else None

def extract_beta(name: str):
    m = re.search(r"beta([0-9]+(?:\.[0-9]+)?)", name)
    return float(m.group(1)) if m else None

def get_N_beta(spins_dir):
    """
    Infer N and beta from available files (patterns preferred).
    """
    files = list(spins_dir.glob("*.csv"))

    N = None
    beta = None

    for f in files:
        if N is None:
            N = extract_N(f.name)
        if beta is None:
            beta = extract_beta(f.name)
        if N is not None and beta is not None:
            break

    return N, beta

# ============================================================
# BIT-SAFE OVERLAP
# ============================================================

def overlap_binary(config_row, pattern_row, N: int) -> float:
    """Compute m = (1/N) sum_i s_i xi_i from 64-bit packed blocks."""
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
# RESOLVE ALPHA SELECTION
# ============================================================

ALL_ALPHA_DIRS = sorted(
    root_dir.glob("alpha_*"),
    key=lambda p: float(p.name.split("_")[1])
)

print(f"\n[INFO] {len(ALL_ALPHA_DIRS)} alpha folder(s) found:")
for i, d in enumerate(ALL_ALPHA_DIRS, start=1):
    print(f"  [{i}] {d.name}")

if ALPHA_SELECTION:
    selected = [
        ALL_ALPHA_DIRS[i - 1]
        for i in ALPHA_SELECTION
        if 1 <= i <= len(ALL_ALPHA_DIRS)
    ]
else:
    selected = ALL_ALPHA_DIRS

print(f"\n[INFO] Processing {len(selected)} alpha folder(s):")
for d in selected:
    print(f"  {d.name}")

# ============================================================
# MAIN LOOP
# ============================================================

for alpha_dir in selected:

    spins_dir = alpha_dir / "spins"
    patterns_dir = alpha_dir / "patterns"

    if not spins_dir.exists() or not patterns_dir.exists():
        print(f"\n[SKIP] {alpha_dir.name}: missing spins/ or patterns/")
        continue

    # -------------------------
    # infer N and beta globally
    # -------------------------
    N, beta = get_N_beta(spins_dir)

    if N is None or beta is None:
        print(f"\n[SKIP] {alpha_dir.name}: cannot infer N/beta")
        continue

    out_dir = alpha_dir / f"overlaps_N{N}_beta{beta}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[ALPHA] {alpha_dir.name} -> overlaps_N{N}_beta{beta}")

    # -------------------------
    # map patterns by realization
    # -------------------------
    pattern_map = {}
    for pf in patterns_dir.glob("*.csv"):
        r = extract_r(pf.name)
        if r is not None:
            pattern_map[r] = pf

    if not pattern_map:
        print(f"[SKIP] {alpha_dir.name}: no patterns found")
        continue

    # -------------------------
    # loop spins
    # -------------------------
    for spin_file in sorted(spins_dir.glob("*.csv")):

        r = extract_r(spin_file.name)
        if r is None or r not in pattern_map:
            continue

        pattern_file = pattern_map[r]

        out_file = out_dir / f"overlaps_N{N}_beta{beta}_r{r}.csv"

        if SKIP_EXISTING and out_file.exists():
            print(f"  r={r} (N={N}, beta={beta}) ... skipped")
            continue

        print(f"  r={r} (N={N}, beta={beta}) ... ", end="", flush=True)

        spins_df = pd.read_csv(spin_file, dtype=str)
        pattern_df = pd.read_csv(pattern_file, dtype=str)

        sweep_col = spins_df.columns[0]
        sweep_ids = spins_df[sweep_col].astype(int).values
        block_cols = [c for c in spins_df.columns if c != sweep_col]

        n_sweeps = len(spins_df)
        n_patterns = len(pattern_df)

        M = np.empty((n_sweeps, n_patterns), dtype=float)

        for i, (_, spin_row) in enumerate(spins_df[block_cols].iterrows()):
            for j, (_, pat_row) in enumerate(pattern_df.iterrows()):
                M[i, j] = overlap_binary(spin_row, pat_row, N)

        cols = ["sweep"] + [f"m_{i}" for i in range(n_patterns)]

        df = pd.DataFrame(
            np.column_stack([sweep_ids, M]),
            columns=cols
        )

        df["sweep"] = df["sweep"].astype(int)
        df.to_csv(out_file, index=False)

        print(f"saved ({n_sweeps} x {n_patterns})")

print("\n[DONE]")