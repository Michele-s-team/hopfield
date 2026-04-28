import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
L = 100
T = 0.5
path = "../results/spins/spin_config_r39.csv"
start_snapshot = 0
end_snapshot = 300
step = 1

def load_snapshots(filename, start, end, step=1):
    snapshots = []
    with open(filename) as f:
        lines = [line.strip() for line in f if line.strip()]
    all_blocks = []
    for s in range(0, len(lines), L):
        block = lines[s:s+L]
        if len(block) == L:
            all_blocks.append(
                np.array([list(map(int, row.split(','))) for row in block])
            )
    return all_blocks[start:end:step]

snapshots = load_snapshots(path, start_snapshot, end_snapshot, step=step)
print(f"{len(snapshots)} snapshots (T={T}, step={step})")

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(snapshots[0], cmap='gray', vmin=-1, vmax=1)
title = ax.set_title(f"T={T:.3f} | Snapshot {start_snapshot + 1} / {start_snapshot + len(snapshots)}")
ax.axis('off')
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(1)
ax.axis('on')
ax.set_xticks([])
ax.set_yticks([])

def update(frame):
    im.set_data(snapshots[frame])
    title.set_text(f"T={T:.1f} | Snapshot {start_snapshot + frame + 1} / {start_snapshot + len(snapshots)}")
    return [im, title]

ani = animation.FuncAnimation(
    fig, update,
    frames=len(snapshots),
    interval=200,
    blit=False
)

# Sauvegarde
output = f"../results/spins/spins_T={T:.1f}.mp4"
ani.save(output, writer='ffmpeg', fps=10, dpi=150)
print(f"video saved : {output}")

plt.tight_layout()
plt.show()