import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


plt.rcParams.update({
    "text.usetex"                : True,
    "font.family"                : "serif",
    "font.serif"                 : ["Computer Modern Roman"],
    "axes.formatter.use_mathtext": True
    })

# ── Data ────────────────────────────────────────────────────────────────────
# NB: adapte le chemin si besoin
df = pd.read_csv("../results/Hopfield/speedup_save_analysis.csv")

# ── Couleurs distinctes par N ────────────────────────────────────────────────
N_vals  = sorted(df["N"].unique())
palette = plt.cm.plasma(np.linspace(0.1, 0.85, len(N_vals)))
color   = {N: palette[i] for i, N in enumerate(N_vals)}

markers = ["o", "s", "D", "^", "v", "P", "X"]
marker  = {N: markers[i % len(markers)] for i, N in enumerate(N_vals)}

# ── Couleurs distinctes par alpha (pour la figure 5) ────────────────────────
alpha_vals   = sorted(df["alpha"].unique())
palette_a    = plt.cm.viridis(np.linspace(0.1, 0.85, len(alpha_vals)))
color_alpha  = {a: palette_a[i] for i, a in enumerate(alpha_vals)}
marker_alpha = {a: markers[i % len(markers)] for i, a in enumerate(alpha_vals)}

# ── Style global ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "xtick.direction"            : "in",
    "ytick.direction"            : "in",
    "xtick.top"                  : True,
    "ytick.right"                : True,
    "xtick.major.size"           : 5,
    "ytick.major.size"           : 5,
    "xtick.minor.size"           : 3,
    "ytick.minor.size"           : 3,
    "xtick.major.width"          : 0.8,
    "ytick.major.width"          : 0.8,
    "xtick.minor.visible"        : True,
    "ytick.minor.visible"        : True,
    "axes.linewidth"             : 0.8,
    "axes.spines.top"            : True,
    "axes.spines.right"          : True,
    "legend.frameon"             : False,
    "legend.fontsize"            : 8,
    "legend.handlelength"        : 2.0,
    "legend.labelspacing"        : 0.3,
    "legend.handletextpad"       : 0.5,
    "font.size"                  : 10,
    "axes.labelsize"             : 11,
    "xtick.labelsize"            : 9,
    "ytick.labelsize"            : 9,
    "lines.linewidth"            : 1.2,
})

BOTTOM = 0.15  # marge inférieure commune à toutes les figures

# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Courbes speed-up vs T, moyennées sur alpha, pour chaque N
# ════════════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(4, 3.2))

for N in N_vals:
    group = (
        df[df["N"] == N]
        .groupby("T")["ratio"]
        .mean()
        .reset_index()
        .sort_values("T")
    )
    ax.plot(
        group["T"], group["ratio"],
        marker=marker[N],
        markersize=4,
        color=color[N],
        linewidth=1.2,
        label=f"$N={N}$",
    )

ax.set_xlabel(r"$T\,/\,J$")
ax.set_ylabel("Speed-up factor")
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(True, which="major", alpha=0.25, linewidth=0.5)
ax.legend(framealpha=0.9, edgecolor="0.7", handlelength=1.8, ncol=2)

fig2.tight_layout()
fig2.subplots_adjust(bottom=BOTTOM)
fig2.savefig("../results/Hopfield/speedup_curves_hopfield.pdf", pad_inches=0.05)
fig2.savefig("../results/Hopfield/speedup_curves_hopfield.png", dpi=300)
print("Saved: speedup_curves_hopfield.pdf / .png")


N_sweeps = 2**16

# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — Temps normalisé t / (N * N_sweeps) vs T (moyenné sur alpha et N)
# ════════════════════════════════════════════════════════════════════════════
fig3, ax1 = plt.subplots(figsize=(4, 3.2))
ax2 = ax1.twinx()

df["t_bits_norm"]   = df["t_bits"]   / (df["N"] * N_sweeps) * 1e9
df["t_nobits_norm"] = df["t_nobits"] / (df["N"] * N_sweeps) * 1e6

mean_bits   = df.groupby("T")["t_bits_norm"].mean()
mean_nobits = df.groupby("T")["t_nobits_norm"].mean()

ax1.plot(mean_bits.index,   mean_bits.values,
         color="C0", linewidth=1.2, linestyle="-",  marker="o", markersize=4, label="bitwise")
ax2.plot(mean_nobits.index, mean_nobits.values,
         color="#2ca02c", linewidth=1.2, linestyle="--", marker="s", markersize=4, label="classic")

ax1.set_xlabel(r"$T\,/\,J$")
ax1.set_ylabel(r"$\langle t_{\rm bits} / (N \cdot N_{\rm sweep}) \rangle$ [ns]",   color="C0")
ax2.set_ylabel(r"$\langle t_{\rm classic} / (N \cdot N_{\rm sweep}) \rangle$ [$\mu$s]", color="#2ca02c")
ax1.tick_params(axis="y", labelcolor="C0")
ax2.tick_params(axis="y", labelcolor="#2ca02c")
ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune=None))
ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune=None))

ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.grid(True, which="major", alpha=0.25, linewidth=0.5)

lines = [plt.Line2D([0],[0], color="C0",      linestyle="-",  marker="o", markersize=4),
         plt.Line2D([0],[0], color="#2ca02c",  linestyle="--", marker="s", markersize=4)]
ax1.legend(lines, ["bitwise", "classic"], fontsize=8, loc="upper right")

fig3.tight_layout()
fig3.subplots_adjust(bottom=BOTTOM)
fig3.savefig("../results/Hopfield/time_normalized_hopfield.pdf", pad_inches=0.05)
fig3.savefig("../results/Hopfield/time_normalized_hopfield.png", dpi=300)
print("Saved: time_normalized_hopfield.pdf / .png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — Temps total vs N (moyenné sur alpha) pour toutes les températures
# ════════════════════════════════════════════════════════════════════════════
T_vals   = sorted(df["T"].unique())
palette4 = plt.cm.coolwarm(np.linspace(0.05, 0.95, len(T_vals)))
color4   = {T: palette4[i] for i, T in enumerate(T_vals)}
 
fig4, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(7, 3.2), sharey=False)
 
for T in T_vals:
    group = (
        df[df["T"] == T]
        .groupby("N")[["t_bits", "t_nobits"]]
        .mean()
        .reset_index()
        .sort_values("N")
    )
    ax_b.plot(group["N"], group["t_bits"]   / N_sweeps * 1e9,
              marker="o", markersize=3, linewidth=1.0, color=color4[T])
    ax_c.plot(group["N"], group["t_nobits"] / N_sweeps * 1e6,
              marker="s", markersize=3, linewidth=1.0, color=color4[T])
 
for ax, col, scale in zip([ax_b, ax_c], ["t_bits", "t_nobits"], [1e9, 1e6]):
    ax.set_xlabel(r"$N$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_locator(ticker.LogLocator(subs="all", numticks=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs="all", numticks=10))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, which="major", alpha=0.25, linewidth=0.5)
 
    # fit log-log (moyenné sur alpha et T)
    all_N = df.groupby("N")[col].mean().index.values
    all_t = df.groupby("N")[col].mean().values / N_sweeps * scale
    slope, intercept = np.polyfit(np.log(all_N), np.log(all_t), 1)
 
    # droite de régression exacte (même pente ET même ordonnée à l'origine
    # que le fit), donc bien centrée au milieu des courbes par T
    N_ref = np.array([df["N"].min(), df["N"].max()])
    t_fit = np.exp(intercept) * N_ref**slope
    ax.plot(N_ref, t_fit,
            color="k", linewidth=0.8, linestyle=":",
            label=rf"$ \propto N^{{{slope:.2f}}}$")
    ax.legend(fontsize=7, frameon=False)
 
ax_b.set_title("bitwise", fontsize=10)
ax_c.set_title("classic", fontsize=10)
ax_b.set_ylabel(r"$t_{\rm bits} / N_{\rm sweep}$ [ns]")
ax_c.set_ylabel(r"$t_{\rm classic} / N_{\rm sweep}$ [$\mu$s]")
 
handles = [plt.Line2D([0],[0], color=color4[T], linewidth=1.0,
                       marker="o", markersize=3, label=f"$T={T:.2f}$")
           for T in T_vals]
fig4.legend(handles=handles, loc="center right", bbox_to_anchor=(1.15, 0.5),
            fontsize=7, frameon=False, title=r"$T/J$", title_fontsize=8)
 
fig4.tight_layout()
fig4.savefig("../results/Hopfield/time_total_vs_N_hopfield.pdf", bbox_inches="tight", pad_inches=0.05)
fig4.savefig("../results/Hopfield/time_total_vs_N_hopfield.png", bbox_inches="tight", dpi=300)
print("Saved: time_total_vs_N_hopfield.pdf / .png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 5 — Speed-up vs T pour chaque alpha, moyenné sur N
# ════════════════════════════════════════════════════════════════════════════
fig5, ax = plt.subplots(figsize=(4, 3.2))

for a in alpha_vals:
    group = (
        df[df["alpha"] == a]
        .groupby("T")["ratio"]
        .mean()
        .reset_index()
        .sort_values("T")
    )
    ax.plot(
        group["T"], group["ratio"],
        marker=marker_alpha[a],
        markersize=4,
        color=color_alpha[a],
        linewidth=1.2,
        label=rf"$\alpha={a:.2f}$",
    )

ax.set_xlabel(r"$T\,/\,J$")
ax.set_ylabel("Speed-up factor")
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(True, which="major", alpha=0.25, linewidth=0.5)
ax.legend(framealpha=0.9, edgecolor="0.7", handlelength=1.8)

fig5.tight_layout()
fig5.subplots_adjust(bottom=BOTTOM)
fig5.savefig("../results/Hopfield/speedup_curves_by_alpha_hopfield.pdf", pad_inches=0.05)
fig5.savefig("../results/Hopfield/speedup_curves_by_alpha_hopfield.png", dpi=300)
print("Saved: speedup_curves_by_alpha_hopfield.pdf / .png")