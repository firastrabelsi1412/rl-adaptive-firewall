"""Generate docs/images/architecture.png (run from anywhere: python docs/make_architecture.py)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "architecture.png")

W, H = 16.0, 11.0  # inches; at dpi=100 -> 1600 x 1100 px
BOX_W, BOX_H = 10.0, 1.9
CX = W / 2

BLUES = ("#e8f1fb", "#2b6cb0")
GREENS = ("#e9f7ef", "#2f855a")
ORANGE = ("#fff4e5", "#c05621")
GREY = ("#f2f2f2", "#8a8a8a")

boxes = [
    dict(y=8.95, title="Network traffic",
         lines=["41 features per flow  ·  windows of 10 records"], colors=BLUES),
    dict(y=6.35, title="LSTM-CNN detector",
         lines=["CNN: spatial patterns  ·  LSTM: temporal patterns",
                "→ 5 classes: Normal · DoS · Probe · R2L · U2R"], colors=GREENS),
    dict(y=3.75, title="PPO agent (Stable-Baselines3)",
         lines=["actions: ALLOW · BLOCK · RATE-LIMIT · LOG"], colors=ORANGE),
    dict(y=1.15, title="iptables on gateway",
         lines=["Phase 5 — in progress"], colors=GREY, dashed=True),
]

fig, ax = plt.subplots(figsize=(W, H), dpi=100)
fig.patch.set_facecolor("white")
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

for b in boxes:
    fill, edge = b["colors"]
    dashed = b.get("dashed", False)
    ax.add_patch(FancyBboxPatch(
        (CX - BOX_W / 2, b["y"]), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.25",
        facecolor=fill, edgecolor=edge, linewidth=2.5,
        linestyle=(0, (6, 4)) if dashed else "-"))
    text_col = "#6b6b6b" if dashed else "#1a202c"
    ax.text(CX, b["y"] + BOX_H - 0.5, b["title"], ha="center", va="center",
            fontsize=24, fontweight="bold", color=text_col)
    for k, line in enumerate(b["lines"]):
        ax.text(CX, b["y"] + BOX_H - 1.1 - 0.45 * k, line, ha="center", va="center",
                fontsize=17, color="#4a5568" if not dashed else "#7a7a7a",
                style="italic" if dashed else "normal")

for upper, lower in zip(boxes[:-1], boxes[1:]):
    dashed = lower.get("dashed", False)
    ax.add_patch(FancyArrowPatch(
        (CX, upper["y"] - 0.05), (CX, lower["y"] + BOX_H + 0.05),
        arrowstyle="-|>", mutation_scale=35, linewidth=2.5,
        color="#8a8a8a" if dashed else "#2d3748",
        linestyle=(0, (4, 3)) if dashed else "-"))

fig.savefig(OUT, dpi=100, facecolor="white", bbox_inches=None)
print("wrote", OUT)
