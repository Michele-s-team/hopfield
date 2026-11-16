#!/usr/bin/env python3
"""
Reads <mode>/corr/q_pairs_alpha*_N*_beta*_.csv for mode in {bits, nobits}.

Writes exactly TWO output files:
    corr_summary_bits.csv
    corr_summary_nobits.csv

Each file contains all (alpha, N, beta) combinations,
with one row per sweep.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

root_dir = Path("../../results/Hopfield/indep/")
MODES = ["bits", "nobits"]
CORR_DIRNAME = "corr"
OUT_DIR = root_dir / "corr_summary"
SKIP_EXISTING = False

FNAME_RE = re.compile(r"^q_pairs_alpha_?(.+?)_N(\d+)_(beta.+?)_\.csv$")

def summarize_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    q_abs = np.abs(df.drop(columns="sweep").to_numpy())

    return pd.DataFrame({
        "sweep": df["sweep"].to_numpy(),
        "q_abs_mean": q_abs.mean(axis=1),
        "q_abs_max": q_abs.max(axis=1),
    })

if __name__ == "__main__":

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for mode in MODES:

        out_file = OUT_DIR / f"corr_summary_{mode}.csv"

        if SKIP_EXISTING and out_file.exists():
            print(f"[SKIP] {out_file.name} already exists")
            continue

        files = sorted((root_dir / mode / CORR_DIRNAME).glob("q_pairs_*.csv"))

        summaries = []

        print(f"\n[INFO] Mode {mode}: {len(files)} file(s)")

        for f in files:
            m = FNAME_RE.match(f.name)

            if not m:
                print(f"[SKIP] cannot parse {f.name}")
                continue

            alpha, N, beta = m.group(1), int(m.group(2)), m.group(3).removeprefix("beta")

            summary = summarize_file(f)

            summary.insert(0, "beta", beta)
            summary.insert(0, "N", N)
            summary.insert(0, "alpha", alpha)

            summaries.append(summary)

        if summaries:
            out = pd.concat(summaries, ignore_index=True)
            out.sort_values(["alpha", "N", "beta", "sweep"], inplace=True)
            out.to_csv(out_file, index=False)

            print(f"[SAVED] {out_file}: {len(out)} rows")
