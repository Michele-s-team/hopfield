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
        total = 0
        try:
            for b in range(n_full):
                val = arr[i, b]
                if pd.isna(val) or val is None or str(val).lower() == 'nan':
                    mags[i] = np.nan
                    raise ValueError(f"NaN found in block {b} at sweep {i}")
                total += int(val).bit_count()
            
            if remainder:
                x_val = arr[i, n_full]
                if pd.isna(x_val) or x_val is None or str(x_val).lower() == 'nan':
                    mags[i] = np.nan
                    raise ValueError(f"NaN found in remainder block at sweep {i}")
                x = int(x_val)
                total += (x & ((1 << remainder) - 1)).bit_count()

            mags[i] = (2 * total - N) / N
        except (ValueError, TypeError) as e:
            mags[i] = np.nan
            continue

    return mags


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

def load_blocks(path):
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as e:
        print(f"    Error reading CSV file {path}: {e}")
        return None, None
    
    if df.empty:
        print(f"    Warning: Empty CSV file: {path}")
        return None, None
    
    if df.shape[1] < 2:
        print(f"    Warning: Not enough columns in {path}: {df.shape[1]}")
        return None, None
    
    try:
        sweep_col = df.iloc[:, 0].astype(int).to_numpy()
        blocks_df = df.iloc[:, 1:]
    except Exception as e:
        print(f"    Error converting data in {path}: {e}")
        return None, None
    
    return sweep_col, blocks_df


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

        T_match = re.search(r"T_([0-9.]+)", os.path.basename(T_dir))
        if not T_match:
            print(f"Warning: Could not extract T from {T_dir}")
            continue
        
        T = T_match.group(1)

        # Vérifier si le fichier de sortie existe déjà
        out_file = os.path.join(out_N_dir, f"N{N}_T{T}.csv")
        
        if os.path.exists(out_file):
            try:
                existing_df = pd.read_csv(out_file)
                if existing_df.empty:
                    print(f"\nT = {T} - Existing file is empty, will overwrite")
                    os.remove(out_file)
                else:
                    print(f"\nT = {T} - File already exists and is valid, skipped")
                    continue
            except Exception as e:
                print(f"\nT = {T} - Existing file is corrupted ({e}), will overwrite")
                os.remove(out_file)
        
        print(f"\nProcessing T = {T}...")

        spin_files = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(T_dir)
            for f in files
            if f.startswith("spins_r") and f.endswith(".csv")
        )

        if not spin_files:
            print(f"  Warning: No spin files found for T={T}, skipping")
            continue

        # ─────────────────────────────────────────
        # D'abord, on vérifie si UNE réalisation a des NaN
        # Si oui, on skip TOUTE la température
        # ─────────────────────────────────────────

        has_nan = False
        problematic_file = None

        for f in spin_files:
            try:
                _, blocks = load_blocks(f)
                if blocks is None:
                    has_nan = True
                    problematic_file = f
                    break
                
                m = magnetization_from_blocks(blocks, N)
                
                # Vérifier s'il y a des NaN dans la magnétisation
                if np.isnan(m).any():
                    has_nan = True
                    problematic_file = f
                    break
                    
            except Exception as e:
                has_nan = True
                problematic_file = f
                print(f"    Error processing {f}: {e}")
                break

        # Si on a trouvé des NaN, on skip toute la température
        if has_nan:
            print(f"  ERROR: NaN detected in realization {os.path.basename(problematic_file)}")
            print(f"  Skipping ALL realizations for T={T} (corrupted data)")
            print(f"  No output file will be created for this temperature")
            continue

        # ─────────────────────────────────────────
        # Si on arrive ici, toutes les réalisations sont valides
        # On peut calculer les moments
        # ─────────────────────────────────────────

        data = []
        m1_list, m2_list, m4_list = [], [], []

        for f in spin_files:
            m_r = re.search(r"r(\d+)", os.path.basename(f))
            r = int(m_r.group(1)) if m_r else -1

            _, blocks = load_blocks(f)
            m = magnetization_from_blocks(blocks, N)

            m1 = np.mean(m)
            m2 = np.mean(m**2)
            m4 = np.mean(m**4)

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

        if R == 0:
            print(f"  ERROR: No valid realizations for T={T}")
            continue

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
        df = df[["r", "m1", "m2", "m4", "dm1", "dm2", "dm4"]]

        if len(df) > 0:
            df.to_csv(out_file, index=False)
            print(f"  Saved: {out_file}")
            print(f"    Realizations: {R}")

print("\nDONE")