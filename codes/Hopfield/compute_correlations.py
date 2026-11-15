#!/usr/bin/env python3
"""
For each mode in {bits, nobits} and each alpha_* / N4096 / beta* directory:
  - reads beta*/spins/spins_r{r}.csv for ALL replicas r
  - computes the bit-safe overlap between every pair of replicas (r < s)
    at the same sweep:  q_rs(t) = (1/N) sum_i s_i^r(t) s_i^s(t)
  - saves ONE file per (mode, alpha, N, beta):
        beta*/overlaps_replicas/q_pairs.csv

Output DataFrame columns: sweep | q_0_1 | q_0_2 | ... | q_{R-2}_{R-1}

bits and nobits are processed independently (never compared to each other).
Existing files are skipped (set SKIP_EXISTING = False to overwrite).
"""

from pathlib import Path
from functools import reduce
import pandas as pd
import numpy as np
import re

# ============================================================
# CONFIGURATION
# ============================================================

# folder that contains bits/ and nobits/  (adjust to your layout)
root_dir = Path("../results/Hopfield/indep/")

MODES = ["bits", "nobits"]

N_TARGET = 4096
ALPHA_SELECTION = None
SKIP_EXISTING = True


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


def corelations_matrix(spins_arr_1: np.ndarray, spins_arr_2: np.ndarray, N: int) -> np.ndarray:
    """
    spins_arr_1: (n_sweeps, n_blocks) uint64   (replica r)
    spins_arr_2: (n_sweeps, n_blocks) uint64   (replica s)

    Returns (n_sweeps,) float array q_rs(t) = (1/N) sum_i s_i^r(t) s_i^s(t),
    computed block by block (64 spins per uint64) with XOR + popcount.
    """
    n_full    = N // 64
    remainder = N % 64

    corr = np.zeros(spins_arr_1.shape[0], dtype=np.int64)

    for b in range(n_full):
        xor = np.bitwise_xor(spins_arr_1[:, b], spins_arr_2[:, b])
        corr += 64 - 2 * popcount64(xor)

    if remainder:
        mask = np.uint64((1 << remainder) - 1)
        xor = np.bitwise_xor(spins_arr_1[:, n_full], spins_arr_2[:, n_full]) & mask
        corr += remainder - 2 * popcount64(xor)

    return corr / N


# ============================================================
# CSV READING (uint64-safe, single parse pass)
# ============================================================

def read_blocks_csv(path: Path):
    """
    Reads a CSV with an id column (sweep) followed by block{i} columns,
    returning (ids, block_array) where block_array is a 2D uint64 NumPy
    array (n_rows, n_blocks).
    """
    df = pd.read_csv(path, dtype=str)
    id_col = df.columns[0]
    block_cols = [c for c in df.columns if c != id_col]

    ids = df[id_col].astype(np.int64).to_numpy()
    block_arr = df[block_cols].to_numpy(dtype=np.uint64)

    return ids, block_arr


# ============================================================
# MAIN LOOP
# ============================================================

for mode in MODES:

    mode_dir = root_dir / mode

    ALL_ALPHA_DIRS = sorted(
        mode_dir.glob("alpha_*"),
        key=lambda p: float(p.name.split("_")[1])
    )

    print(f"\n[INFO] {mode}/: {len(ALL_ALPHA_DIRS)} alpha folder(s) found:")
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

    for alpha_dir in selected:

        alpha = float(alpha_dir.name.split("_")[1])
        N_dir = alpha_dir / f"N{N_TARGET}"
        N = N_TARGET

        if not N_dir.exists():
            print(f"\n[SKIP] {mode}/{alpha_dir.name}: no N{N_TARGET} folder")
            continue

        beta_dirs = sorted(
            N_dir.glob("beta*"),
            key=lambda p: float(p.name.split("beta")[1])
        )

        if not beta_dirs:
            print(f"\n[SKIP] {N_dir}: no beta* folder found")
            continue

        for beta_dir in beta_dirs:

            beta_str = beta_dir.name.replace("beta", "")
            spin_dir = beta_dir / "spins"

            if not spin_dir.exists():
                print(f"  [SKIP] missing spins/ in {mode}/{beta_dir.name}")
                continue

            spin_map = {}
            for sf in spin_dir.glob("spins_r*.csv"):
                r = extract_r(sf.name)
                if r is not None:
                    spin_map[r] = sf

            replicas = sorted(spin_map)
            if len(replicas) < 2:
                print(f"  [SKIP] fewer than 2 replicas in {mode}/{beta_dir.name}")
                continue

            out_dir = beta_dir / "overlaps_replicas"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "q_pairs.csv"

            print(f"\n[{mode.upper()}] alpha={alpha:.4f} N={N} beta={beta_str} -> {out_dir.name}")

            if SKIP_EXISTING and out_file.exists():
                print("  skipped")
                continue

            # read every replica once
            sweeps_all = {}
            spins_all = {}
            for r in replicas:
                print(f"  reading r={r}", end=" ... ", flush=True)
                sweeps_all[r], spins_all[r] = read_blocks_csv(spin_map[r])
                print("done")

            # keep only the sweeps present in all replicas
            common_sweeps = reduce(np.intersect1d, sweeps_all.values())
            idx = {
                r: np.searchsorted(sweeps_all[r], common_sweeps)
                if np.all(np.diff(sweeps_all[r]) > 0)
                else np.argsort(sweeps_all[r])[np.searchsorted(np.sort(sweeps_all[r]), common_sweeps)]
                for r in replicas
            }
            spins_sel = {r: spins_all[r][idx[r]] for r in replicas}

            # overlap of every pair of replicas (r < s)
            columns = {}
            for i, r in enumerate(replicas):
                for s in replicas[i + 1:]:
                    columns[f"q_{r}_{s}"] = corelations_matrix(
                        spins_sel[r], spins_sel[s], N
                    )

            df = pd.DataFrame(columns)
            df.insert(0, "sweep", common_sweeps.astype(int))
            df.to_csv(out_file, index=False)

            print(f"saved ({len(df)} sweeps x {len(columns)} pairs)")