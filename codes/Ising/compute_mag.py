import numpy as np
import glob
import os
import re
import pandas as pd

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

base_dir = "../results/Ising"
output_dir = "../results/Ising/moments"
os.makedirs(output_dir, exist_ok=True)

# ─────────────────────────────────────────────
# MAGNETIZATION
# ─────────────────────────────────────────────

def magnetization_from_blocks(blocks, N):
    n_full = N // 64
    remainder = N % 64
    arr = blocks.values.astype(object)
    n_sweeps = arr.shape[0]

    mags = np.zeros(n_sweeps, dtype=np.float64)

    for i in range(n_sweeps):
        total = sum(int(arr[i, b]).bit_count() for b in range(n_full))
        if remainder:
            x = int(arr[i, n_full])
            total += (x & ((1 << remainder) - 1)).bit_count()

        mags[i] = (2 * total - N) / N

    return mags


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

def load_blocks(path):
    df = pd.read_csv(path, dtype=str)
    return df.iloc[:, 0].astype(int).to_numpy(), df.iloc[:, 1:]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

N_dirs = sorted(glob.glob(os.path.join(base_dir, "N*")), 
                key=lambda x: int(re.search(r"N(\d+)$", os.path.basename(x)).group(1)))

for N_dir in N_dirs:

    N = int(re.search(r"N(\d+)$", os.path.basename(N_dir)).group(1))

    print("\n" + "=" * 40)
    print(f"N = {N}")

    out_N_dir = os.path.join(output_dir, f"N{N}")
    os.makedirs(out_N_dir, exist_ok=True)

    T_dirs = sorted(glob.glob(os.path.join(N_dir, "T_*")), reverse=True)

    for T_dir in T_dirs:

        T = re.search(r"T_([0-9.]+)", os.path.basename(T_dir)).group(1)

        # Vérifier si le fichier de sortie existe déjà
        out_file = os.path.join(out_N_dir, f"N{N}_T{T}.csv")
        
        if os.path.exists(out_file):
            print(f"\nT = {T} - File already exists, ignored")
            continue
        
        print(f"\nProcessing T = {T}...")

        spin_files = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(T_dir)
            for f in files
            if f.startswith("spins_r") and f.endswith(".csv")
        )

        data = []
        m1_list, m2_list, m4_list = [], [], []

        # ─────────────────────────────
        # per realization
        # ─────────────────────────────

        for f in spin_files:

            m_r = re.search(r"r(\d+)", os.path.basename(f))
            r = int(m_r.group(1)) if m_r else -1

            _, blocks = load_blocks(f)
            m = magnetization_from_blocks(blocks, N)

            m1 = np.mean(m)
            m2 = np.mean(m**2)
            m4 = np.mean(m**4)

            # Pour chaque réalisation, l'erreur est l'écart-type divisé par sqrt(n_sweeps)
            n_sweeps = len(m)
            dm1 = np.std(m, ddof=1) / np.sqrt(n_sweeps) if n_sweeps > 1 else 0.0
            dm2 = np.std(m**2, ddof=1) / np.sqrt(n_sweeps) if n_sweeps > 1 else 0.0
            dm4 = np.std(m**4, ddof=1) / np.sqrt(n_sweeps) if n_sweeps > 1 else 0.0

            data.append({
                "r": r,
                "m1": m1,
                "m2": m2,
                "m4": m4,
                "dm1": dm1,
                "dm2": dm2,
                "dm4": dm4
            })

            m1_list.append(m1)
            m2_list.append(m2)
            m4_list.append(m4)

        R = len(data)

        def err(x):
            return np.std(x, ddof=1) / np.sqrt(R) if R > 1 else 0.0

        mean_row = {
            "r": "mean",
            "m1": np.mean(m1_list),
            "m2": np.mean(m2_list),
            "m4": np.mean(m4_list),
            "dm1": err(m1_list),
            "dm2": err(m2_list),
            "dm4": err(m4_list),
        }

        df = pd.DataFrame([mean_row] + data)

        # force column order explicitly
        df = df[["r", "m1", "m2", "m4", "dm1", "dm2", "dm4"]]

        df.to_csv(out_file, index=False)

        print("saved:", out_file)

print("\nDONE")