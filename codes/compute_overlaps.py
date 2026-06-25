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

root_dir = Path("../results")

# -- Sélection des alphas ------------------------------------
# Liste les dossiers disponibles et affiche leur index au lancement.
# Laisser vide [] pour tout traiter.
# Exemples :
#   ALPHA_SELECTION = []          # tous
#   ALPHA_SELECTION = [1]         # premier alpha uniquement
#   ALPHA_SELECTION = [1, 3, 5]   # alphas d'index 1, 3 et 5

ALPHA_SELECTION = [1, 2, 3, 4]

SKIP_EXISTING = True   # False → écrase les fichiers déjà calculés

# ============================================================
# HELPERS
# ============================================================

def extract_r(name: str):
    m = re.search(r"_r(\d+)\.csv$", name)
    return int(m.group(1)) if m else None

def extract_N(name: str):
    m = re.search(r"N(\d+)", name)
    return int(m.group(1)) if m else None

# ============================================================
# BIT-SAFE OVERLAP
# ============================================================

def overlap_binary(config_row, pattern_row, N: int) -> float:
    """Compute m = (1/N) sum_i s_i xi_i from 64-bit packed blocks."""
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
# DETECT N
# ============================================================

any_pattern = next(root_dir.rglob("patterns_N*.csv"))
N = extract_N(any_pattern.name)
print(f"[INFO] N = {N}")

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

    spins_dir    = alpha_dir / "spins"
    patterns_dir = alpha_dir / "patterns"

    if not spins_dir.exists() or not patterns_dir.exists():
        print(f"\n[SKIP] {alpha_dir.name}: missing spins/ or patterns/")
        continue

    alpha   = float(alpha_dir.name.split("_")[1])
    out_dir = alpha_dir / "overlaps"
    out_dir.mkdir(exist_ok=True)
    print(f"\n[ALPHA] {alpha:.4f}")

    pattern_map = {}
    for pf in patterns_dir.glob("*.csv"):
        r = extract_r(pf.name)
        if r is not None:
            pattern_map[r] = pf

    for spin_file in sorted(spins_dir.glob("*.csv")):

        r = extract_r(spin_file.name)
        if r is None or r not in pattern_map:
            continue

        out_file = out_dir / f"overlaps_r{r}.csv"

        if SKIP_EXISTING and out_file.exists():
            print(f"  r={r} ... skipped (already exists)")
            continue

        print(f"  r={r}", end=" ... ", flush=True)

        spins_df   = pd.read_csv(spin_file,      dtype=str)
        pattern_df = pd.read_csv(pattern_map[r], dtype=str)

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
        df   = pd.DataFrame(np.column_stack([sweep_ids, M]), columns=cols)
        df["sweep"] = df["sweep"].astype(int)

        df.to_csv(out_file, index=False)
        print(f"saved ({n_sweeps} sweeps x {n_patterns} patterns)")

print("\n[DONE]")