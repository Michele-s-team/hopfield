import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(15, 11))
ax.set_xlim(0, 150)
ax.set_ylim(0, 110)
ax.axis('off')

# ===============================================================
# PARAMÈTRES GLOBAUX : AJUSTEMENT DES TAILLES
# ===============================================================
FONT_SIZE_BASE = 12.5  # Modifie cette valeur pour agrandir/réduire tout le texte
BOX_W = 20
BOX_H = 10
# ===============================================================

# Color palette
COL_UTIL   = '#EDE7F6'  # bitwise utility classes (Bits, UnsignedInt)
COL_INFRA  = '#E3F2FD'  # infra / helper classes (Metropolis, SimulationIO)
COL_BASE   = '#FFF3E0'  # SpinSystem / SimulationBase backbone
COL_MODEL  = '#E8F5E9'  # abstract *Model classes
COL_BITS   = '#FCE4EC'  # *Bits concrete implementations
COL_NOBITS = '#E0F2F1'  # *NoBits concrete implementations
EDGE       = '#37474F'

def box(x, y, text, color, bold_title=None, w=BOX_W, h=BOX_H):
    b = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.2,rounding_size=1.0",
                        linewidth=1.3, edgecolor=EDGE, facecolor=color, zorder=2)
    ax.add_patch(b)
    if bold_title:
        # Titre de la classe
        ax.text(x + w/2, y + h - 2.0, bold_title, ha='center', va='top',
                 fontsize=FONT_SIZE_BASE+2, fontweight='bold', zorder=3, color='#1a1a1a')
        
        # Gestion spécifique si la description contient du LaTeX ET un retour à la ligne
        if "$" in text and "\n" in text:
            parts = text.split("\n", 1)
            math_part = parts[0].strip()
            text_part = parts[1].strip()
            
            # Première ligne (Hamiltonien) : affichage plus grand
            ax.text(x + w/2, y + h/2 - 1.0, math_part, ha='center', va='center',
                     fontsize=FONT_SIZE_BASE+1, zorder=3, color='#333333')
            # Deuxième ligne (Description standard) : affichage normal
            ax.text(x + w/2, y + h/2 - 3.2, text_part, ha='center', va='center',
                     fontsize=FONT_SIZE_BASE - 2.0, zorder=3, color='#333333', linespacing=1.3)
        else:
            # Affichage standard uniforme si pas de formule mixte
            ax.text(x + w/2, y + h/2 - 2.2, text, ha='center', va='center',
                     fontsize=FONT_SIZE_BASE - 2.0, zorder=3, color='#333333', linespacing=1.3)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                 fontsize=FONT_SIZE_BASE, zorder=3, color='#1a1a1a')
    return (x, y, w, h)

def center_top(b):
    return (b[0] + b[2]/2, b[1] + b[3])

def center_bottom(b):
    return (b[0] + b[2]/2, b[1])

def center_left(b):
    return (b[0], b[1] + b[3]/2)

def center_right(b):
    return (b[0] + b[2], b[1] + b[3]/2)

def arrow_inherit(b_child, b_parent):
    """Open-triangle UML inheritance arrow: child -> parent"""
    p1 = center_top(b_child)
    p2 = center_bottom(b_parent)
    arr = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=14,
                           color=EDGE, linewidth=1.2, zorder=1,
                           shrinkA=2, shrinkB=2)
    ax.add_patch(arr)

def arrow_compose(p1, p2, color='#616161', style='->', ls='dashed'):
    arr = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12,
                           color=color, linewidth=1.0, zorder=1,
                           linestyle=ls, shrinkA=2, shrinkB=2)
    ax.add_patch(arr)

# Alignement parfait sur l'axe vertical central x = 75
X_CENTER = 75 - (BOX_W / 2)

# ---------------------------------------------------------------
# Row 0 : SpinSystem (rehaussé à y=96)
# ---------------------------------------------------------------
b_spinsys = box(X_CENTER, 96, "Builds the lattice \n (PBC/OBC/FC/random)", COL_BASE, bold_title="SpinSystem")

# ---------------------------------------------------------------
# Row 1 : Metropolis | SimulationBase | SimulationIO (rehaussé à y=82)
# ---------------------------------------------------------------
row1_y = 82
b_metro   = box(X_CENTER - 32, row1_y, "RNG / Dynamics", COL_INFRA, bold_title="Metropolis")
b_simbase = box(X_CENTER,      row1_y, "Bridges SpinSystem with \n Metropolis dynamics \n and I/O", COL_BASE, bold_title="SimulationBase")
b_io      = box(X_CENTER + 32, row1_y, "Open/Close files", COL_INFRA, bold_title="SimulationIO")

# Inheritance arrows: SimulationBase -> SpinSystem
arrow_inherit(b_simbase, b_spinsys)

# Composition arrows from SimBase to Metropolis and IO
arrow_compose(center_left(b_simbase), center_right(b_metro), color='#616161', style='-|>', ls='dashed')
arrow_compose(center_right(b_simbase), center_left(b_io), color='#616161', style='-|>', ls='dashed')

# ---------------------------------------------------------------
# Row 2 : abstract *Model classes (rehaussé à y=66)
# ---------------------------------------------------------------
row2_y = 66
b_ising_m = box(X_CENTER - 45, row2_y, r"$H_\text{Ising}$" + "\n ", COL_MODEL, bold_title="IsingModel")
b_sg_m    = box(X_CENTER,      row2_y, r"$H_\text{SG}$" + "\n Couplings", COL_MODEL, bold_title="SpinGlassModel")
b_hop_m   = box(X_CENTER + 45, row2_y, r"$H_\text{Hopfield}$" + "\n Patterns, Couplings", COL_MODEL, bold_title="HopfieldModel")

arrow_inherit(b_ising_m, b_simbase)
arrow_inherit(b_sg_m, b_simbase)
arrow_inherit(b_hop_m, b_simbase)

# ---------------------------------------------------------------
# Row 3 : concrete Bits / NoBits leaves (rehaussé à y=50)
# ---------------------------------------------------------------
row3_y = 50
b_ising_nobits = box(X_CENTER - 56, row3_y, "standard implementation", COL_NOBITS, bold_title="IsingNoBits")
b_ising_bits   = box(X_CENTER - 34, row3_y, "bitwise implementation", COL_BITS, bold_title="IsingBits")

b_sg_nobits    = box(X_CENTER - 11, row3_y, "standard implementation", COL_NOBITS, bold_title="SpinGlassNoBits")
b_sg_bits      = box(X_CENTER + 11, row3_y, "bitwise implementation", COL_BITS, bold_title="SpinGlassBits")

b_hop_nobits   = box(X_CENTER + 34, row3_y, "standard implementation", COL_NOBITS, bold_title="HopfieldNoBits")
b_hop_bits     = box(X_CENTER + 56, row3_y, "bitwise implementation", COL_BITS, bold_title="HopfieldBits")

for child, parent in [(b_ising_nobits, b_ising_m), (b_ising_bits, b_ising_m),
                       (b_sg_nobits, b_sg_m), (b_sg_bits, b_sg_m),
                       (b_hop_nobits, b_hop_m), (b_hop_bits, b_hop_m)]:
    arrow_inherit(child, parent)

# ---------------------------------------------------------------
# Row 4 : UnsignedInt and Bits (rehaussés respectivement à y=34 et y=20)
# ---------------------------------------------------------------
b_uint = box(X_CENTER, 34, "extends Bits", COL_UTIL, bold_title="UnsignedInt")
b_bits = box(X_CENTER, 20, "bitwise operations", COL_UTIL, bold_title="Bits")

# UnsignedInt above Bits connection
arrow_compose(center_bottom(b_uint) , center_top(b_bits), color='#5E35B1', style='-|>', ls='solid')

# dashed "uses" arrows from Bits up to the *Bits leaves
for target in [b_ising_bits, b_sg_bits, b_hop_bits]:
    arrow_compose(center_bottom(target), center_top(b_uint), color='#9575CD', style='-|>', ls='dashed')

# ---------------------------------------------------------------
# Legend (Intact, reste à sa place d'origine)
# ---------------------------------------------------------------
legend_items = [
    (COL_UTIL,  "Bitwise arithmetic"),
    (COL_INFRA, "Composed helpers"),
    (COL_BASE,  "Backbone"),
    (COL_MODEL, "Abstract Hamiltonian layer"),
    (COL_BITS,  "Bitwise-parallel implementation"),
    (COL_NOBITS,"Classical reference implementation"),
]

LEGEND_W = BOX_W * 0.2  
LEGEND_H = BOX_W * 0.2  
FONT_SIZE_LEGEND = FONT_SIZE_BASE - 1

legend_y = 9.0  
for i, (col, label) in enumerate(legend_items):
    x0 = 6 + (i % 3) * 48
    y0 = legend_y - (i // 3) * 5.0
    ax.add_patch(FancyBboxPatch((x0, y0), LEGEND_W, LEGEND_H, boxstyle="round,pad=0.1",
                                 linewidth=1, edgecolor=EDGE, facecolor=col, zorder=2))
    ax.text(x0 + LEGEND_W + 1.5, y0 + LEGEND_H/2, label, fontsize=FONT_SIZE_LEGEND, va='center', color='#1a1a1a')

# Arrow legends (Intact)
arrow_legend_y = 2.5
ax.add_patch(FancyArrowPatch((6, arrow_legend_y), (14, arrow_legend_y), arrowstyle='-|>', mutation_scale=14, color=EDGE, linewidth=1.2))
ax.text(15.5, arrow_legend_y, "public inheritance", fontsize=FONT_SIZE_LEGEND, va='center')

ax.add_patch(FancyArrowPatch((54, arrow_legend_y), (62, arrow_legend_y), arrowstyle='-|>', mutation_scale=12, color='#616161', linewidth=1.0, linestyle='dashed'))
ax.text(63.5, arrow_legend_y, "composition (has-a)", fontsize=FONT_SIZE_LEGEND, va='center')

ax.add_patch(FancyArrowPatch((102, arrow_legend_y), (110, arrow_legend_y), arrowstyle='-|>', mutation_scale=12, color='#9575CD', linewidth=1.0, linestyle='dashed'))
ax.text(111.5, arrow_legend_y, "uses (bitwise engine)", fontsize=FONT_SIZE_LEGEND, va='center')

plt.tight_layout()
plt.savefig('architecture.pdf', format='pdf', bbox_inches='tight')
print("done")