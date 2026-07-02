#!/usr/bin/env python3
"""
For each selected alpha_* / N{N} directory:
  - reads pattern/patterns_r{r}.csv
  - reads beta{beta}_spin/spins_r{r}.csv
  - computes the bit-safe overlap of every sweep with every pattern
  - saves to N{N}/overlaps_beta{beta}/overlaps_r{r}.csv

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

root_dir = Path("../results/Hopfield")

ALPHA_SELECTION = None
SKIP_EXISTING = True


# ============================================================
# HELPERS
# ============================================================

def extract_r(name: str):
    m = re.search(r"_r(\d+)\.csv$", name)
    return int(m.group(1)) if m else None


# beta5.000000 -> "5.000000"
def extract_beta_from_dir(name: str):
    m = re.search(r"^beta([\d.]+)$", name)
    return m.group(1) if m else None


# ============================================================
# BIT-SAFE OVERLAP
# ============================================================

def overlap_binary(config_row, pattern_row, N: int) -> float:
    n_full    = N // 64
    remainder = N % 64
    corr      = 0

    for b in range(n_full):
        s   = int(str(config_row[f"block{b}"]))
        p   = int(str(pattern_row[f"block{b}"]))
        xor = s ^ p
        corr += 64 - 2 * xor.bit_count()

    if remainder:
        s    = int(str(config_row[f"block{n_full}"]))
        p    = int(str(pattern_row[f"block{n_full}"]))
        mask = (1 << remainder) - 1
        xor  = (s ^ p) & mask
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

    alpha = float(alpha_dir.name.split("_")[1])

    N_dirs = sorted(
        alpha_dir.glob("N*"),
        key=lambda p: int(re.search(r"N(\d+)$", p.name).group(1))
    )

    if not N_dirs:
        print(f"\n[SKIP] {alpha_dir.name}: no N* folder found")
        continue

    for N_dir in N_dirs:

        m_N = re.search(r"N(\d+)$", N_dir.name)
        if not m_N:
            print(f"  [SKIP] cannot parse N from {N_dir.name}")
            continue

        N = int(m_N.group(1))

        beta_dirs = sorted(
            N_dir.glob("beta*"),
            key=lambda p: float(p.name.split("beta")[1])
        )

        if not beta_dirs:
            print(f"\n[SKIP] {N_dir.name}: no beta* folder found")
            continue

        for beta_dir in beta_dirs:

            beta_str = beta_dir.name.replace("beta", "")

            pattern_dir = beta_dir / "patterns"
            spin_dir    = beta_dir / "spins"

            if not pattern_dir.exists():
                print(f"  [SKIP] missing patterns/ in {beta_dir.name}")
                continue

            if not spin_dir.exists():
                print(f"  [SKIP] missing spins/ in {beta_dir.name}")
                continue

            pattern_map = {}
            for pf in pattern_dir.glob("patterns_r*.csv"):
                r = extract_r(pf.name)
                if r is not None:
                    pattern_map[r] = pf

            out_dir = beta_dir / f"overlaps"
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[ALPHA] {alpha:.4f} N={N} beta={beta_str} -> {out_dir.name}")

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

                spins_df   = pd.read_csv(spin_file, dtype=str)
                pattern_df = pd.read_csv(pattern_file, dtype=str)

                sweep_col  = spins_df.columns[0]
                sweep_ids  = spins_df[sweep_col].astype(int).values
                block_cols = [c for c in spins_df.columns if c != sweep_col]

                n_sweeps   = len(spins_df)
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

                print("saved")
                print(f"saved ({n_sweeps} sweeps x {n_patterns} patterns)")