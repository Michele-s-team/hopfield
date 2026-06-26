import numpy as np
import glob
import os
import re
import pathlib
import pandas as pd

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
base_dir   = "../results/Ising"
output_dir = "../results/Ising/magnetization_matrix"

os.makedirs(output_dir, exist_ok=True)


# ─────────────────────────────────────────────
# MAGNETIZATION
# ─────────────────────────────────────────────

def magnetization_from_blocks(blocks, N):
    n_full    = N // 64
    remainder = N % 64
    arr       = blocks.values.astype(object)  # garde les grands entiers
    n_sweeps  = arr.shape[0]
    mags      = np.zeros(n_sweeps, dtype=np.float64)

    for i in range(n_sweeps):
        total = sum(int(arr[i, b]).bit_count() for b in range(n_full))
        if remainder:
            x     = int(arr[i, n_full])
            total += (x & ((1 << remainder) - 1)).bit_count()
        mags[i] = (2 * total - N) / N

    return mags

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

def load_blocks(path):
    df = pd.read_csv(path, dtype=str)
    sweeps = df.iloc[:, 0].astype(int).to_numpy()
    blocks = df.iloc[:, 1:]
    return sweeps, blocks

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

T_dirs = sorted(glob.glob(os.path.join(base_dir, "T_*")), reverse=True)

for T_dir in T_dirs:

    match = re.search(r"T_([0-9.]+)", os.path.basename(T_dir))
    if not match:
        continue
    T = match.group(1)

    print("\n==============================")
    print(f"Processing T = {T}")

    # infer N from spins_N*_beta* subdir name
    spins_subdirs = list(pathlib.Path(T_dir).glob("spins_N*_beta*"))
    if not spins_subdirs:
        print("  ERROR: no spins_N*_beta* subdir found")
        continue
    m_N = re.search(r"spins_N(\d+)_beta", spins_subdirs[0].name)
    if not m_N:
        print("  ERROR: cannot extract N from subdir name")
        continue
    N = int(m_N.group(1))
    print(f"  N = {N}")

    spin_files = []
    for root, _, files in os.walk(T_dir):
        for f in files:
            if f.startswith("spins_r") and f.endswith(".csv"):
                spin_files.append(os.path.join(root, f))
    spin_files = sorted(spin_files)

    print(f"  found files: {len(spin_files)}")
    if len(spin_files) == 0:
        continue

    out_file = os.path.join(output_dir, f"T_{T}.csv")
    if os.path.exists(out_file):
        print(f"  skipped (already exists): {out_file}")
        continue

    all_m      = []
    r_list     = []
    sweeps_ref = None
    bad_r      = []

    for f in spin_files:

        m_r = re.search(r"r(\d+)", os.path.basename(f))
        
        if not m_r:
            print(f"[SKIP] no r found in {f}")
            continue
        r = int(m_r.group(1))
        print(f"  r={r:<5}", end="\r", flush=True)

        sweeps, blocks = load_blocks(f)
        if sweeps is None or blocks is None:
            print(f"[ERROR] load failed | r={r}")
            bad_r.append(r)
            continue

        if blocks.shape[1] != (N // 64 + (1 if N % 64 else 0)):
            print(f"[ERROR] block size mismatch | r={r} | shape={blocks.shape}")
            bad_r.append(r)
            continue

        try:
            m = magnetization_from_blocks(blocks, N)
        except Exception as e:
            print(f"[ERROR] magnetization failed | r={r} | {e}")
            bad_r.append(r)
            continue

        if sweeps_ref is None:
            sweeps_ref = sweeps
        elif len(sweeps_ref) != len(sweeps):
            print(f"[WARNING] sweep mismatch | r={r}")

        all_m.append(m)
        r_list.append(r)

    if len(all_m) == 0:
        print(f"  empty T = {T}")
        continue

    if bad_r:
        print(f"  WARNING bad r: {sorted(set(bad_r))}")

    M     = np.stack(all_m, axis=1)
    order = np.argsort(r_list)
    M     = M[:, order]
    r_sorted = sorted(r_list)

    with open(out_file, "w") as out:
        out.write("sweep," + ",".join([f"r{r}" for r in r_sorted]) + "\n")
        for i, s in enumerate(sweeps_ref):
            out.write(
                str(s) + "," +
                ",".join(str(M[i, j]) for j in range(M.shape[1])) +
                "\n"
            )

    print("  saved:", out_file)

print("\nDONE")