#!/usr/bin/env python3

import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ============================================================
# UTIL
# ============================================================

def extract_N(name):
    m = re.search(r"N(\d+)", name)
    return int(m.group(1)) if m else None


def num_blocks(N):
    return (N + 63) // 64


def decode_row_to_spins(row, N, n_blocks):
    """Deterministic decoding: NEVER uses float."""
    remainder = N % 64
    n_full = N // 64

    spins = np.empty(N, dtype=np.int8)
    idx = 0

    for b in range(n_full):
        val = int(str(row[f"block{b}"]))
        for bit in range(64):
            spins[idx] = 1 if (val >> bit) & 1 else -1
            idx += 1

    if remainder:
        val = int(str(row[f"block{n_full}"]))
        for bit in range(remainder):
            spins[idx] = 1 if (val >> bit) & 1 else -1
            idx += 1

    return spins


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    root_dir = Path("../results")

    realization = 17

    spins_dir = root_dir / "spins"
    patterns_dir = root_dir / "patterns"

    spins_file = next(spins_dir.glob(f"*_r{realization}.csv"))
    pattern_file = next(patterns_dir.glob(f"*_r{realization}.csv"))

    print("SPINS FILE   =", spins_file.resolve())
    print("PATTERN FILE =", pattern_file.resolve())

    N = extract_N(spins_file.name)
    L = int(round(np.sqrt(N)))
    if L * L != N:
        raise ValueError("N is not a perfect square")

    n_blocks = num_blocks(N)

    config_df = pd.read_csv(spins_file, dtype=str)
    pattern_df = pd.read_csv(pattern_file, dtype=str)

    pattern_row = pattern_df.iloc[0]

    pattern = decode_row_to_spins(pattern_row, N, n_blocks).reshape(L, L)

    sweeps = config_df["sweep"].astype(int).to_numpy()
    n_frames = len(config_df)

    spin_frames = np.empty((n_frames, L, L), dtype=np.int8)

    for k in range(n_frames):
        spin_frames[k] = decode_row_to_spins(config_df.iloc[k], N, n_blocks).reshape(L, L)

    # ============================================================
    # DEBUG (minimal)
    # ============================================================

    final_spins = spin_frames[-1]

    print("\nCONSISTENCY CHECK")
    print("equal:", np.array_equal(final_spins, pattern))
    print("diffs:", np.count_nonzero(final_spins != pattern))

    # ============================================================
    # PLOT + ANIMATION
    # ============================================================

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))

    # LEFT: animation
    im_left = ax_left.imshow(
        spin_frames[0],
        cmap="binary",
        vmin=-1,
        vmax=1,
        interpolation="nearest"
    )
    title_left = ax_left.set_title(f"Spins (sweep {sweeps[0]})")

    ax_left.set_xticks([])
    ax_left.set_yticks([])

    # RIGHT: static pattern
    im_right = ax_right.imshow(
        pattern,
        cmap="binary",
        vmin=-1,
        vmax=1,
        interpolation="nearest"
    )
    ax_right.set_title("Pattern")
    ax_right.set_xticks([])
    ax_right.set_yticks([])

    fig.suptitle(f"realization r={realization}")

    def update(i):
        im_left.set_data(spin_frames[i])
        title_left.set_text(f"sweep {sweeps[i]}")
        return (im_left, title_left)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=200,
        blit=False,
        repeat=True
    )

    plt.show()