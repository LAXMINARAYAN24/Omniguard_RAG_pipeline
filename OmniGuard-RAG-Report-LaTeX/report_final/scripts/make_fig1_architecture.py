import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY = "#1F3864"
SLATE = "#44546A"
NAVY_LIGHT = "#DBE9F6"
WHITE = "#FFFFFF"
GOLD = "#B08D57"

fig, ax = plt.subplots(figsize=(9.5, 11.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 128)
ax.axis("off")

def box(x, y, w, h, text, fc=NAVY_LIGHT, ec=NAVY, tc=NAVY, fs=9.3, bold=True, radius=2.4, lw=1.4):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={radius}",
                        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
             fontsize=fs, fontweight="bold" if bold else "normal", zorder=4, linespacing=1.35)
    return (x + w / 2, y, x + w / 2, y + h, x, y + h / 2, x + w, y + h / 2)

def arrow(p1, p2, color=SLATE, lw=1.6, style="-|>", connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                         color=color, linewidth=lw, zorder=2,
                         connectionstyle=connectionstyle, shrinkA=2, shrinkB=2)
    ax.add_patch(a)

# Title
ax.text(50, 125.5, "OmniGuard-RAG \u2014 System Architecture", ha="center", va="center",
        fontsize=15, fontweight="bold", color=NAVY)
ax.text(50, 122.3, "Rings 0\u20133 and the Dynamic Trust Store", ha="center", va="center",
        fontsize=10, color=SLATE, style="italic")

# Row: Query / Documents inputs
ax.text(22, 116.5, "Raw User Query  q", ha="center", fontsize=9.7, color=SLATE, fontweight="bold")
ax.text(78, 116.5, "Candidate Documents  D", ha="center", fontsize=9.7, color=SLATE, fontweight="bold")
arrow((22, 116.0), (22, 113.4))
arrow((78, 116.0), (78, 113.4))

# Ring 0 / Ring 1
b_r0 = box(6, 104, 32, 9, "Ring 0: Query-Path Guard\nSuffix repetition-ratio\nscreen (PIDP defense)", fc=NAVY_LIGHT, ec=NAVY, tc=NAVY)
b_r1 = box(62, 104, 32, 9, "Ring 1: Spectral Guard\nDRS \u2014 fit/calibration-split\nPCA eigendecomposition", fc=NAVY_LIGHT, ec=NAVY, tc=NAVY)

ax.text(22, 100.6, "Sanitized q", ha="center", fontsize=8.3, color=SLATE, style="italic")
ax.text(78, 100.6, "Verified-clean D", ha="center", fontsize=8.3, color=SLATE, style="italic")
arrow((22, 104), (22, 98.3))
arrow((78, 104), (78, 98.3))
arrow((22, 98.3), (46, 93.6), connectionstyle="arc3,rad=-0.12")
arrow((78, 98.3), (54, 93.6), connectionstyle="arc3,rad=0.12")

# Dense index + trust store
b_idx = box(28, 84, 44, 9.6, "Shared TF-IDF Retrieval Index\n+ Dynamic Trust Store\n(persists across queries)", fc=NAVY, ec=NAVY, tc=WHITE, fs=9.7)
ax.text(50, 81.6, "Top-k retrieval", ha="center", fontsize=8.3, color=SLATE, style="italic")
arrow((50, 84), (50, 78.6))

# Ring 2 router
b_r2 = box(20, 66, 60, 12.2,
           "Ring 2: Risk-Aware Router\nTWO independent signals \u2014 embedding cohesion  AND  answer-vote contention.\nEscalate to Ring 3 if EITHER signal fires.",
           fc=NAVY_LIGHT, ec=NAVY, tc=NAVY, fs=9.4)

ax.text(28, 62.0, "Low risk", ha="center", fontsize=8.6, color=SLATE, fontweight="bold")
ax.text(72, 62.0, "High risk (either signal)", ha="center", fontsize=8.6, color=SLATE, fontweight="bold")
arrow((32, 66), (28, 58.4), connectionstyle="arc3,rad=-0.08")
arrow((68, 66), (72, 58.4), connectionstyle="arc3,rad=0.08")

# Fast path / Ring 3
b_fast = box(6, 46, 34, 10.8, "Fast Path\nWeighted-majority vote,\nsingle pass (~1x compute)", fc=WHITE, ec=SLATE, tc=SLATE, fs=9.2)
b_r3 = box(56, 46, 38, 10.8, "Ring 3: GWCC\nLeave-one-out +\nclique-restricted leave-pair-out", fc=NAVY_LIGHT, ec=NAVY, tc=NAVY, fs=9.2)

arrow((23, 46), (44, 40.2), connectionstyle="arc3,rad=-0.12")
arrow((75, 46), (56, 40.2), connectionstyle="arc3,rad=0.12")

# Trust store update
b_ts = box(28, 30, 44, 9.0, "Dynamic Trust Store Update\nDown-weights documents implicated\nby Ring 3; feeds later queries", fc=GOLD, ec="#8A6D3B", tc=WHITE, fs=8.9)
arrow((50, 30), (50, 24.6))

b_out = box(30, 15, 40, 8.4, "Trusted Response", fc=NAVY, ec=NAVY, tc=WHITE, fs=11)

# Feedback loop from trust store back to index (dashed)
fb = FancyArrowPatch((28, 34.5), (28, 88.8), arrowstyle="-|>", mutation_scale=12,
                      color=GOLD, linewidth=1.3, linestyle=(0, (4, 3)), zorder=1,
                      connectionstyle="arc3,rad=0.35")
ax.add_patch(fb)
ax.text(9.5, 61, "reweights\nretrieval in\nlater queries", ha="center", fontsize=7.4, color="#8A6D3B",
        style="italic", rotation=90, linespacing=1.2)

plt.tight_layout()
plt.savefig("/home/claude/report_build/figures/fig1_architecture.png", dpi=220, bbox_inches="tight", facecolor="white")
print("saved fig1")
