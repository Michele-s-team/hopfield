import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys

path = "../results/spins/config_spins_r3.csv"
start_snapshot = 0
end_snapshot = 200
T = 0.05
L = 50

with open(path) as f:
    lines = [line.strip() for line in f if line.strip()]


def load_snapshots(filename, start, end):
    snapshots = []
    with open(filename) as f:
        lines = [line.strip() for line in f if line.strip()]
    for s in range(0, len(lines), L):
        block = lines[s:s+L]
        if len(block) == L:
            snapshots.append(
                np.array([list(map(int, row.split(','))) for row in block])
            )
    return snapshots[start:end]

snapshots = load_snapshots(path, start_snapshot, end_snapshot)
print(f"{len(snapshots)} snapshots (T={T})")

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(snapshots[0], cmap='gray', vmin=-1, vmax=1)
title = ax.set_title(f"T={T:.3f} | Snapshot {start_snapshot + 1} / {start_snapshot + len(snapshots)}")
ax.axis('off')

def update(frame):
    im.set_data(snapshots[frame])
    title.set_text(f"T={T:.3f} | Snapshot {start_snapshot + frame + 1} / {start_snapshot + len(snapshots)}")
    return [im, title]

ani = animation.FuncAnimation(
    fig, update,
    frames=len(snapshots),
    interval=200,
    blit=False
)

# Sauvegarde
output = f"../results/spins/spins_T={T:.2f}.mp4"
ani.save(output, writer='ffmpeg', fps=10, dpi=150)
print(f"video saved : {output}")

plt.tight_layout()
plt.show()