#!/usr/bin/env python3
"""
For each mode in {bits, nobits} and each alpha_* / N4096 / beta* directory:
  - looks at beta*/spins/spins_r*.csv for ALL replicas
  - removes the very last line (duplicated sweep) of each file, in place

A file is only trimmed if its last two lines share the same sweep index
(first field), so running the script twice is harmless.
Set FORCE = True to remove the last line unconditionally.
Set DRY_RUN = True to only report what would be done.
"""

from pathlib import Path
import time

# ============================================================
# CONFIGURATION
# ============================================================

root_dir = Path("../../results/Hopfield/indep/")

MODES = ["bits", "nobits"]
N_TARGET = 4096
ALPHA_SELECTION = None

DRY_RUN = False      # mets False pour modifier réellement les fichiers
FORCE = False       # True : supprime la dernière ligne sans vérifier le doublon

BLOCK = 1 << 20     # 1 MiB


# ============================================================
# HELPERS
# ============================================================

def newline_positions_from_end(f, end: int, count: int) -> list:
    """Offsets of the last `count` newlines located before offset `end`."""
    found, pos = [], end
    while pos > 0 and len(found) < count:
        start = max(0, pos - BLOCK)
        f.seek(start)
        chunk = f.read(pos - start)
        i = len(chunk)
        while len(found) < count:
            i = chunk.rfind(b"\n", 0, i)
            if i == -1:
                break
            found.append(start + i)
        pos = start
    return found


def trim_last_line(path: Path) -> str:
    with open(path, "rb+") as f:
        size = f.seek(0, 2)
        if size == 0:
            return "empty file"

        # ignore the trailing newline, if any
        f.seek(size - 1)
        end = size - 1 if f.read(1) == b"\n" else size

        nl = newline_positions_from_end(f, end, 2)
        if not nl:
            return "single line, untouched"

        last_start = nl[0] + 1
        if len(nl) < 2:
            # only header + one data line: nothing to compare against
            if not FORCE:
                return "only one data line, untouched"
            prev_start = None
        else:
            prev_start = nl[1] + 1

        if not FORCE:
            f.seek(last_start)
            last = f.read(end - last_start).decode().strip()
            f.seek(prev_start)
            prev = f.read(nl[0] - prev_start).decode().strip()
            if last.split(",", 1)[0] != prev.split(",", 1)[0]:
                return "last sweep not duplicated, untouched"

        if not DRY_RUN:
            f.truncate(last_start)
        return "last line removed" + (" (dry run)" if DRY_RUN else "")


# ============================================================
# BUILD FILE LIST
# ============================================================

def build_files():
    files = []
    for mode in MODES:
        mode_dir = root_dir / mode

        all_alpha_dirs = sorted(
            mode_dir.glob("alpha_*"),
            key=lambda p: float(p.name.split("_")[1])
        )

        print(f"[INFO] {mode}/: {len(all_alpha_dirs)} alpha folder(s) found:")
        for i, d in enumerate(all_alpha_dirs, start=1):
            print(f"  [{i}] {d.name}")

        if ALPHA_SELECTION:
            selected = [
                all_alpha_dirs[i - 1]
                for i in ALPHA_SELECTION
                if 1 <= i <= len(all_alpha_dirs)
            ]
        else:
            selected = all_alpha_dirs

        for alpha_dir in selected:
            N_dir = alpha_dir / f"N{N_TARGET}"
            if not N_dir.exists():
                print(f"[SKIP] {mode}/{alpha_dir.name}: no N{N_TARGET} folder")
                continue

            beta_dirs = sorted(
                N_dir.glob("beta*"),
                key=lambda p: float(p.name.split("beta")[1])
            )
            if not beta_dirs:
                print(f"[SKIP] {N_dir}: no beta* folder found")
                continue

            for beta_dir in beta_dirs:
                spin_dir = beta_dir / "spins"
                if not spin_dir.exists():
                    print(f"[SKIP] {beta_dir}: missing spins/")
                    continue
                files.extend(sorted(spin_dir.glob("spins_r*.csv")))

    return files


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    files = build_files()
    print(f"\n[INFO] {len(files)} spins file(s)"
          f"{' (DRY RUN)' if DRY_RUN else ''}\n", flush=True)

    t_start = time.time()

    for k, f in enumerate(files, start=1):
        rel = f.relative_to(root_dir)
        print(f"({k}/{len(files)}) {rel}: {trim_last_line(f)}", flush=True)

    print(f"\n[INFO] done in {time.time() - t_start:.1f}s")