import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import os
import re

# ── Style ─────────────────────────────────────────────
mpl.rcParams.update({
    "text.usetex"            : True,
    "font.family"            : "serif",
    "font.serif"             : ["Computer Modern Roman"],
    "axes.formatter.use_mathtext": True,

    "xtick.direction"        : "in",
    "ytick.direction"        : "in",
    "xtick.top"              : True,
    "ytick.right"            : True,
    "xtick.major.size"       : 5,
    "ytick.major.size"       : 5,
    "xtick.minor.size"       : 3,
    "ytick.minor.size"       : 3,
    "xtick.major.width"      : 0.8,
    "ytick.major.width"      : 0.8,
    "xtick.minor.visible"    : True,
    "ytick.minor.visible"    : True,
    "axes.linewidth"         : 0.8,
    "axes.spines.top"        : True,
    "axes.spines.right"      : True,

    "legend.frameon"         : False,
    "legend.fontsize"        : 8,
    "legend.handlelength"    : 2.0,
    "legend.labelspacing"    : 0.3,
    "legend.handletextpad"   : 0.5,

    "font.size"              : 10,
    "axes.labelsize"         : 11,
    "xtick.labelsize"        : 9,
    "ytick.labelsize"        : 9,
    "lines.linewidth"        : 1.2,
})

mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

# ── Palette : marqueurs creux, couleurs vives ──────────
COLORS  = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#e07b00', '#8c564b']
MARKERS = ['o', 's', '^', 'D', 'v', 'p']
# fillstyle='none' → marqueurs creux

N_LIST = [900, 2500, 6400, 10000, 22500, 40000]
L_LIST = [30,  50,   80,  100,   150,   200  ]

moments_dir = "../results/Ising/moments"
output_dir  = "../results/Ising"
os.makedirs(output_dir, exist_ok=True)

# ── Load ───────────────────────────────────────────────
def load_file(fpath):
    df = pd.read_csv(fpath)
    df = df[df["r"].astype(str) != "mean"].copy()
    df["r"] = df["r"].astype(int)
    return df

print(f"Recherche dans: {os.path.abspath(moments_dir)}")
if not os.path.exists(moments_dir):
    raise ValueError(f"Répertoire introuvable: {moments_dir}")

N_dirs = sorted(glob.glob(os.path.join(moments_dir, "N*")))
print(f"Dossiers N: {[os.path.basename(d) for d in N_dirs]}")
if not N_dirs:
    raise ValueError("Aucun dossier N* trouvé")

data = {}
for N_dir in N_dirs:
    m = re.search(r"N(\d+)$", os.path.basename(N_dir))
    if not m:
        continue
    N = int(m.group(1))
    csv_files = sorted(glob.glob(os.path.join(N_dir, "*.csv")))
    if not csv_files:
        continue
    T_dict = {}
    for f in csv_files:
        mt = re.search(r"T_?([0-9.]+?)(?:\.csv)", os.path.basename(f))
        if not mt:
            continue
        try:
            T  = float(mt.group(1))
            df = load_file(f)
            if len(df) > 0:
                T_dict[T] = df
        except Exception as e:
            print(f"  [ERR] {os.path.basename(f)}: {e}")
    if T_dict:
        data[N] = T_dict
        print(f"  N={N}: {len(T_dict)} T, {len(next(iter(T_dict.values())))} réalisations")

if not data:
    raise ValueError("Aucune donnée chargée")

N_avail = [N for N in N_LIST if N in data]
L_avail = [L_LIST[N_LIST.index(N)] for N in N_avail]
colors  = {N: COLORS[N_LIST.index(N)] for N in N_avail}
markers = {N: MARKERS[N_LIST.index(N)] for N in N_avail}

# ── Onsager ────────────────────────────────────────────
Tc   = 2.0 / np.log(1.0 + np.sqrt(2.0))
T_th = np.linspace(0.5, 3.1, 2000)
arg  = 1 - np.sinh(2 / T_th) ** (-4)
m_th = np.where(arg > 0, arg ** (1/8), 0.0)

# ── Bootstrap U4 ───────────────────────────────────────
def binder_and_err(m2, m4, n_boot=500, seed=42):
    m2_mean = np.mean(m2)
    m4_mean = np.mean(m4)
    denom   = 3 * m2_mean ** 2
    U4      = 1 - m4_mean / denom if denom > 0 else np.nan
    rng     = np.random.default_rng(seed)
    idx     = rng.integers(0, len(m2), size=(n_boot, len(m2)))
    u4_b    = 1 - np.mean(m4[idx], axis=1) / (3 * np.mean(m2[idx], axis=1) ** 2)
    return U4, np.std(u4_b, ddof=1)

ew = 0.6
cs = 2
ms = 5.0

def plot_series(ax, N, L, T_vals, y_vals, err_vals):
    c  = colors[N]
    mk = markers[N]
    ax.plot(T_vals, y_vals,
            color=c, lw=1.2, zorder=2)        # lignes sous Onsager (zorder=3)
    ax.errorbar(T_vals, y_vals, yerr=err_vals,
                fmt=mk, color=c,
                ms=ms, lw=0,
                elinewidth=ew, capsize=cs, capthick=ew,
                fillstyle='none',
                markeredgewidth=1.0,
                zorder=4,                      # marqueurs au-dessus de tout
                label=rf"$L={L}$")

# ══════════════════════════════════════════════════════
# PLOT 1 : sqrt(<m²>) vs T
# ══════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(3.4, 2.8))

for N, L in zip(N_avail, L_avail):
    T_vals = np.array(sorted(data[N].keys()))
    rms    = np.zeros(len(T_vals))
    err    = np.zeros(len(T_vals))
    for i, T in enumerate(T_vals):
        m2     = data[N][T]["m2"].values
        r      = np.sqrt(np.mean(m2))
        std_m2 = np.std(m2, ddof=1)
        rms[i] = r
        err[i] = std_m2 / (2 * r * np.sqrt(len(m2))) if r > 0 else 0.0
    plot_series(ax1, N, L, T_vals, rms, err)

ax1.plot(T_th, m_th, color="black", lw=1, zorder=3, label="Onsager")
ax1.axvline(Tc, color="0.4", lw=0.8, ls="--",  zorder=0)

# ── Axe du haut avec uniquement Tc ────────────────────
ax1_top = ax1.twiny()
ax1_top.set_xlim(ax1.get_xlim())  # sync avant set_xticks
ax1_top.set_xticks([Tc])
ax1_top.set_xticklabels([r"$T_c$"], fontsize=8)
ax1_top.tick_params(direction="in", length=5, width=0.8)
ax1_top.xaxis.set_minor_locator(ticker.NullLocator())

ax1.set_xlim(0.9, 3.1)
ax1_top.set_xlim(0.9, 3.1)  # ← resync après set_xlim sur ax1
ax1.set_ylim(-0.02, 1.05)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax1.set_xlabel(r"$T/J$")
ax1.set_ylabel(r"$\sqrt{\langle m^2 \rangle}$")

# Légende relevée : ancre à y=0.45 pour éviter la zone basse
ax1.legend(loc="lower left", ncol=2, columnspacing=0.8,
           bbox_to_anchor=(0.01, 0.05))

plt.tight_layout(pad=0.3)
fig1.savefig(os.path.join(output_dir, "magnetizations_multiN.pdf"), bbox_inches="tight")
print("Saved: magnetizations_multiN.pdf")
plt.show()

# ══════════════════════════════════════════════════════
# PLOT 2 : Binder cumulant + inset zoom
# ══════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(3.5, 3.2))  # un peu plus haut

U4_all  = {}
err_all = {}

for N, L in zip(N_avail, L_avail):
    T_vals = np.array(sorted(data[N].keys()))
    U4     = np.zeros(len(T_vals))
    err_U4 = np.zeros(len(T_vals))
    for i, T in enumerate(T_vals):
        m2 = data[N][T]["m2"].values
        m4 = data[N][T]["m4"].values
        U4[i], err_U4[i] = binder_and_err(m2, m4)
    U4_all[N]  = (T_vals, U4)
    err_all[N] = err_U4
    plot_series(ax2, N, L, T_vals, U4, err_U4)

ax2.axhline(2/3, color="0.65", lw=0.7, ls=":", zorder=0)
ax2.axhline(0,   color="0.65", lw=0.7, ls=":", zorder=0)
ax2.axvline(Tc,  color="0.6",  lw=0.7, ls=":", zorder=0)

# ── Ticks personnalisés sur y : 0, 2/3, + multiples de 0.1 ──
y_major = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, round(2/3, 10), 0.7]
ax2.set_yticks(y_major)
ax2.set_yticklabels([
    r"$0$", r"$0.1$", r"$0.2$", r"$0.3$",
    r"$0.4$", r"$0.5$", r"$0.6$",
    r"$2/3$", r"$0.7$"
])

# Réduire la taille du tick 2/3 uniquement
for label in ax2.get_yticklabels():
    if "2/3" in label.get_text():
        label.set_fontsize(6.5)
        
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

# ── Axe x du haut avec un seul tick à Tc ──────────────
ax2_top = ax2.twiny()
ax2_top.set_xlim(ax2.get_xlim())
ax2_top.set_xticks([Tc])
ax2_top.set_xticklabels([r"$T_c$"], fontsize=8)
ax2_top.tick_params(direction="in", length=5, width=0.8)
# Supprimer les autres ticks du haut
ax2_top.xaxis.set_minor_locator(ticker.NullLocator())

ax2.set_xlim(0.9, 3.1)
ax2.set_ylim(-0.08, 0.78)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax2.set_xlabel(r"$T/J$")
ax2.set_ylabel(r"$U_4$")

# ── Inset ──────────────────────────────────────────────
ax_in = ax2.inset_axes([0.085, 0.285, 0.50, 0.44])
zoom_xmin, zoom_xmax = 2.2135, 2.325
zoom_ymin, zoom_ymax = 0.53, 0.68

for N, L in zip(N_avail, L_avail):
    T_vals, U4 = U4_all[N]
    err_U4     = err_all[N]
    mask_line  = (T_vals >= zoom_xmin - 0.05) & (T_vals <= zoom_xmax + 0.05)
    mask_pts   = (T_vals >= zoom_xmin)         & (T_vals <= zoom_xmax)
    Tz_line    = T_vals[mask_line]
    U4_line    = U4[mask_line]
    Tz_pts     = T_vals[mask_pts]
    U4z        = U4[mask_pts]
    errz       = err_U4[mask_pts]
    if len(Tz_line) == 0:
        continue
    c  = colors[N]
    mk = markers[N]
    ax_in.plot(Tz_line, U4_line, color=c, lw=0.9, zorder=2)
    if len(Tz_pts) > 0:
        ax_in.errorbar(Tz_pts, U4z, yerr=errz,
                       fmt=mk, color=c,
                       ms=3.5, lw=0,
                       elinewidth=0.5, capsize=1.5, capthick=0.5,
                       fillstyle='none', markeredgewidth=0.8,
                       zorder=3)

ax_in.axvline(Tc, color="0.6", lw=0.6, ls=":")
ax_in.set_xlim(zoom_xmin, zoom_xmax)
ax_in.set_ylim(zoom_ymin, zoom_ymax)
ax_in.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax_in.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
ax_in.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax_in.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
ax_in.tick_params(labelsize=6, which="major", length=3, direction="in", pad=1)
ax_in.tick_params(which="minor", length=1.5, direction="in")
for sp in ax_in.spines.values():
    sp.set_linewidth(0.6)

ax2.indicate_inset_zoom(ax_in, edgecolor="0.5", lw=0.6)

# ── Légende en bas à gauche, sous l'inset ─────────────
ax2.legend(loc="lower left", ncol=2, columnspacing=0.8,
           bbox_to_anchor=(0.0, 0.0))

plt.tight_layout(pad=0.3)
fig2.savefig(os.path.join(output_dir, "binder_multiN.pdf"), bbox_inches="tight")
print("Saved: binder_multiN.pdf")
plt.show()