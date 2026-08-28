"""Regenerate Fig. 1 (pipeline.pdf) to match the method as it now stands.

The previous diagram showed an SVD projection feeding a ConvLSTM-EWMA arm and a
filter keyed on the ConvLSTM score. That component was removed from the method,
so the figure described a different system than the paper.

This draws the three stages the paper actually specifies:
   1  SVD latent projection            z_{k,t} = B (delta_{k,t} - mu)
   2  Five-signal statistical screen   r_{k,t} -> EWMA -> rbar_{k,t}
   3  Dual-threshold filter + aggregation

Output: pipeline.pdf (vector) and pipeline.png (preview).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.2,
})

W, H = 7.16, 2.55                       # IEEE double-column width
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

INK    = "#1a1a1a"
CLIENT = "#e8eef7"; CLIENT_E = "#4a6fa5"
MAL    = "#f8e3e3"; MAL_E    = "#b5484a"
STAGE  = "#eef3ea"; STAGE_E  = "#5b7a4a"
AGG    = "#f3eff7"; AGG_E    = "#6b5b95"


def box(x, y, w, h, fc, ec, lw=0.9, r=0.035):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, zorder=2))


def txt(x, y, s, size=7.2, weight="normal", color=INK, ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, zorder=4)


def arrow(x1, y1, x2, y2, style="-|>", lw=0.9, color=INK, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=8, lw=lw, color=color, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}", zorder=3))


# ── clients ────────────────────────────────────────────────────────────────
txt(0.62, 2.34, "Clients", 7.6, "bold")
for i, (lbl, fc, ec) in enumerate([
        (r"$k=1$", CLIENT, CLIENT_E),
        (r"$k=2$", CLIENT, CLIENT_E),
        (r"$\vdots$", CLIENT, CLIENT_E),
        (r"$k\in\mathcal{M}$", MAL, MAL_E)]):
    y = 1.90 - i * 0.44
    box(0.16, y, 0.92, 0.34, fc, ec)
    txt(0.62, y + 0.17, lbl, 7.0)
arrow(1.10, 1.30, 1.52, 1.30)
txt(1.31, 1.46, r"$\hat{\mathbf{g}}_{k,t}$", 7.0)

# ── server box ─────────────────────────────────────────────────────────────
box(1.52, 0.42, 5.42, 1.92, "#fbfbfb", "#9a9a9a", lw=0.8, r=0.02)
txt(1.62, 2.22, "Server", 7.6, "bold", ha="left")

sx, sw, sy, sh = [1.72, 3.52, 5.32], 1.52, 0.86, 1.10

# stage 1
box(sx[0], sy, sw, sh, STAGE, STAGE_E)
txt(sx[0] + sw / 2, sy + sh - 0.17, "1  SVD projection", 7.2, "bold")
txt(sx[0] + sw / 2, sy + 0.60, r"$\mathbf{z}_{k,t}=\mathbf{B}(\boldsymbol{\delta}_{k,t}-\boldsymbol{\mu})$", 7.0)
txt(sx[0] + sw / 2, sy + 0.28, "basis fixed from\nclean warm-up", 6.3, color="#555555")

# stage 2
box(sx[1], sy, sw, sh, STAGE, STAGE_E)
txt(sx[1] + sw / 2, sy + sh - 0.17, "2  Five-signal screen", 7.2, "bold")
txt(sx[1] + sw / 2, sy + 0.66,
    "norm · direction · displacement\nsign · clone", 6.4)
txt(sx[1] + sw / 2, sy + 0.30,
    r"$r_{k,t}\!=\!\sum_i\beta_i r^{(i)}_{k,t}\;\to\;\bar r_{k,t}$", 7.0)

# stage 3
box(sx[2], sy, sw, sh, AGG, AGG_E)
txt(sx[2] + sw / 2, sy + sh - 0.17, "3  Filter + aggregate", 6.9, "bold")
txt(sx[2] + sw / 2, sy + 0.62,
    r"$\tau^{\mathrm{filt}}_t\!=\!\min(\tau^{(q)}_t,\tau_{\mathrm{abs}})$", 7.0)
txt(sx[2] + sw / 2, sy + 0.30,
    r"$\widetilde{S}_t\!=\!\{k:\bar r_{k,t}\!\leq\!\tau^{\mathrm{filt}}_t\}$", 7.0)

for i in (0, 1):
    arrow(sx[i] + sw, sy + sh / 2, sx[i + 1], sy + sh / 2)

# median reference annotation
txt(sx[1] + sw / 2, sy - 0.16,
    r"every signal referenced to the round's coordinate-wise median $\bar{\mathbf{z}}_t$",
    6.4, color="#5b7a4a")

# ── broadcast feedback ─────────────────────────────────────────────────────
arrow(sx[2] + sw / 2, sy, sx[2] + sw / 2, 0.20, style="-")
arrow(sx[2] + sw / 2, 0.20, 0.62, 0.20, style="-")
arrow(0.62, 0.20, 0.62, 0.55, style="-|>")
txt(3.6, 0.10, r"broadcast $\mathbf{w}^{t+1}=\mathrm{AGG}(\widetilde{S}_t)$", 6.6, color="#444444")

fig.savefig("pipeline.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("pipeline.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
print("wrote pipeline.pdf / pipeline.png")
