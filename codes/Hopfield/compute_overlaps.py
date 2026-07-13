#!/usr/bin/env python3
"""
For each selected beta{beta} / N{N} / alpha_* directory:
  - reads patterns/patterns_r{r}.csv
  - reads spins/spins_r{r}.csv
  - computes the bit-safe overlap of every sweep with every pattern
  - saves to alpha_*/overlaps/overlaps_r{r}.csv

Output DataFrame columns: sweep | m_0 | m_1 | ... | m_{p-1}

Existing files are skipped (set SKIP_EXISTING = False to overwrite).

NOTE: overlap computation is fully vectorized with NumPy (no per-element
Python loop), which is the main change vs. the original scalar version.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re

# ============================================================
# CONFIGURATION
# ============================================================

root_dir = Path("../results/Hopfield/phase_transition_multiN_init_from_pattern/")

ALPHA_SELECTION = None
SKIP_EXISTING = True

print(f"[INFO] cwd      : {Path.cwd()}")
print(f"[INFO] root_dir : {root_dir.resolve()}")


# ============================================================
# HELPERS
# ============================================================

def extract_r(name: str):
    m = re.search(r"_r(\d+)\.csv$", name)
    return int(m.group(1)) if m else None


# ============================================================
# VECTORIZED BIT-SAFE OVERLAP
# ============================================================

# 256-entry lookup table for byte-wise popcount, reused for every call
_POPCOUNT_TABLE8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(arr_u64: np.ndarray) -> np.ndarray:
    """
    Vectorized popcount for an array of uint64 values (any shape).
    Views each uint64 as 8 bytes and sums looked-up byte popcounts.
    """
    b = arr_u64.view(np.uint8).reshape(arr_u64.shape + (8,))
    return _POPCOUNT_TABLE8[b].sum(axis=-1, dtype=np.int64)


def overlap_matrix(spins_arr: np.ndarray, patterns_arr: np.ndarray, N: int) -> np.ndarray:
    """
    spins_arr:    (n_sweeps, n_blocks)   uint64
    patterns_arr: (n_patterns, n_blocks) uint64

    Returns (n_sweeps, n_patterns) float overlap matrix, equivalent to
    calling the original scalar overlap_binary() for every (sweep, pattern)
    pair, but computed with a handful of vectorized NumPy operations
    (one per 64-bit block) instead of a Python-level double loop.
    """
    n_full    = N // 64
    remainder = N % 64

    n_sweeps   = spins_arr.shape[0]
    n_patterns = patterns_arr.shape[0]
    corr = np.zeros((n_sweeps, n_patterns), dtype=np.int64)

    for b in range(n_full):
        # outer XOR: shape (n_sweeps, n_patterns)
        xor = np.bitwise_xor.outer(spins_arr[:, b], patterns_arr[:, b])
        corr += 64 - 2 * popcount64(xor)

    if remainder:
        mask = np.uint64((1 << remainder) - 1)
        xor = np.bitwise_xor.outer(spins_arr[:, n_full], patterns_arr[:, n_full]) & mask
        corr += remainder - 2 * popcount64(xor)

    return corr / N


# ============================================================
# CSV READING (uint64-safe, single parse pass)
# ============================================================

def read_blocks_csv(path: Path):
    """
    Reads a CSV with an id column (sweep or pattern index) followed by
    block{i} columns, returning (ids, block_array) where block_array is
    a 2D uint64 NumPy array (n_rows, n_blocks).
    """
    df = pd.read_csv(path, dtype=str)
    id_col = df.columns[0]
    block_cols = [c for c in df.columns if c != id_col]

    ids = df[id_col].astype(np.int64).to_numpy()
    block_arr = df[block_cols].to_numpy(dtype=np.uint64)

    return ids, block_arr


# ============================================================
# RESOLVE BETA DIRECTORIES
# ============================================================

ALL_BETA_DIRS = sorted(
    root_dir.glob("beta*"),
    key=lambda p: float(p.name.replace("beta", ""))
)

print(f"\n[INFO] {len(ALL_BETA_DIRS)} beta folder(s) found:")
for d in ALL_BETA_DIRS:
    print(f"  {d.name}")


# ============================================================
# MAIN LOOP
# ============================================================

for beta_dir in ALL_BETA_DIRS:

    beta_str = beta_dir.name.replace("beta", "")

    N_dirs = sorted(
        beta_dir.glob("N*"),
        key=lambda p: int(re.search(r"N(\d+)$", p.name).group(1))
    )

    if not N_dirs:
        print(f"\n[SKIP] {beta_dir.name}: no N* folder found")
        continue

    for N_dir in N_dirs:

        m_N = re.search(r"N(\d+)$", N_dir.name)
        if not m_N:
            print(f"  [SKIP] cannot parse N from {N_dir.name}")
            continue

        N = int(m_N.group(1))

        alpha_dirs = sorted(
            N_dir.glob("alpha_*"),
            key=lambda p: float(p.name.split("_")[1])
        )

        if not alpha_dirs:
            print(f"\n[SKIP] {N_dir.name}: no alpha_* folder found")
            continue

        if ALPHA_SELECTION:
            alpha_dirs = [
                d for d in alpha_dirs
                if float(d.name.split("_")[1]) in ALPHA_SELECTION
            ]

        for alpha_dir in alpha_dirs:

            alpha = float(alpha_dir.name.split("_")[1])

            pattern_dir = alpha_dir / "patterns"
            spin_dir    = alpha_dir / "spins"

            if not pattern_dir.exists():
                print(f"\n[SKIP] {alpha_dir.name}: missing patterns/")
                continue

            if not spin_dir.exists():
                print(f"\n[SKIP] {alpha_dir.name}: missing spins/")
                continue

            pattern_map = {}
            for pf in pattern_dir.glob("patterns_r*.csv"):
                r = extract_r(pf.name)
                if r is not None:
                    pattern_map[r] = pf

            out_dir = alpha_dir / "overlaps"
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[BETA] {beta_str} N={N} alpha={alpha:.4f} -> {out_dir.name}")

            for spin_file in sorted(spin_dir.glob("spins_r*.csv")):

                r = extract_r(spin_file.name)
                if r is None or r not in pattern_map:
                    continue

                pattern_file = pattern_map[r]
                out_file = out_dir / f"overlaps_r{r}.csv"

                if SKIP_EXISTING and out_file.exists():
                    print(f"  r={r} ... skipped")
                    continue

                print(f"  r={r}", end=" ... ", flush=True)

                sweep_ids, spins_arr    = read_blocks_csv(spin_file)
                _,         patterns_arr = read_blocks_csv(pattern_file)

                M = overlap_matrix(spins_arr, patterns_arr, N)

                n_sweeps, n_patterns = M.shape
                cols = ["sweep"] + [f"m_{i}" for i in range(n_patterns)]

                df = pd.DataFrame(
                    np.column_stack([sweep_ids, M]),
                    columns=cols
                )
                df["sweep"] = df["sweep"].astype(int)
                df.to_csv(out_file, index=False)

                print(f"saved ({n_sweeps} sweeps x {n_patterns} patterns)")