import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY = "#1F3864"
SLATE = "#44546A"
GOLD = "#B08D57"
RED = "#B33A3A"
GREY = "#8A97A8"
LGREY = "#DCE2EA"

plt.rcParams["font.family"] = "DejaVu Sans"

# ---------------------------------------------------------------------------
# FIGURE 2 -- Main system comparison: Accuracy vs Overall ASR, 8 seeds, mean+-95% CI
# Source: results/path_a_report.md, Table 1 (verified by independent single-seed
# reproduction against the delivered code in this session).
# ---------------------------------------------------------------------------
systems = ["Vanilla\nRAG", "DRS\nOnly", "ShieldRAG\nOnly", "RAGuard /\nZKIP", "TriShield", "OmniGuard-\nRAG (Ours)"]
acc_mean = [85.1, 85.6, 85.1, 80.5, 85.5, 100.0]
acc_ci   = [0.1, 0.1, 0.1, 0.4, 0.1, 0.0]
asr_mean = [0.9, 0.2, 0.9, 6.5, 0.2, 0.0]
asr_ci   = [0.2, 0.1, 0.2, 0.5, 0.1, 0.0]

x = np.arange(len(systems))
w = 0.36

fig, ax1 = plt.subplots(figsize=(9.6, 5.4))
ax2 = ax1.twinx()

colors_acc = [SLATE]*5 + [NAVY]
colors_asr = [GREY]*5 + [RED]

b1 = ax1.bar(x - w/2, acc_mean, w, yerr=acc_ci, capsize=3.5, color=colors_acc,
             edgecolor="white", linewidth=0.6, label="Accuracy (%)", zorder=3,
             error_kw={"elinewidth": 1.1, "ecolor": "#222222"})
b2 = ax2.bar(x + w/2, asr_mean, w, yerr=asr_ci, capsize=3.5, color=colors_asr,
             edgecolor="white", linewidth=0.6, label="Overall ASR (%)", zorder=3,
             error_kw={"elinewidth": 1.1, "ecolor": "#222222"})

ax1.set_ylim(0, 108)
ax2.set_ylim(0, 14.2)
ax1.set_ylabel("Accuracy (%)", color=SLATE, fontsize=10.5, fontweight="bold")
ax2.set_ylabel("Overall Attack Success Rate (%)", color=RED, fontsize=10.5, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=SLATE)
ax2.tick_params(axis="y", labelcolor=RED)
ax1.set_xticks(x)
ax1.set_xticklabels(systems, fontsize=9.3)
ax1.grid(axis="y", color=LGREY, linewidth=0.8, zorder=0)
ax1.set_axisbelow(True)

for xi, v, e in zip(x - w/2, acc_mean, acc_ci):
    ax1.text(xi, v + e + 2.0, f"{v:.1f}%", ha="center", fontsize=7.6, color=SLATE, fontweight="bold")
for xi, v, e in zip(x + w/2, asr_mean, asr_ci):
    ax2.text(xi, v + e + 0.25, f"{v:.1f}%", ha="center", fontsize=7.6, color=RED, fontweight="bold")

ax1.set_title("Main System Comparison \u2014 8 independent seeds, 1,600 queries/system\nmean \u00b1 95% CI (Student's-t)",
              fontsize=11.5, color=NAVY, fontweight="bold", pad=14)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
           frameon=False, fontsize=9.3)

plt.tight_layout()
plt.savefig("/home/claude/report_build/figures/fig2_main_comparison.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print("saved fig2")

# ---------------------------------------------------------------------------
# FIGURE 3 -- Per-ring ablation ladder: Stealth-collusion ASR and Accuracy,
# trust store deliberately excluded from all four steps.
# Source: results/path_a_report.md, Table 2.
# ---------------------------------------------------------------------------
steps = ["Ring0\nalone", "+Ring1\n(DRS)", "+Ring2\n(cohesion\nonly)", "+Ring2\n(both\nsignals)"]
stealth_mean = [1.1, 1.1, 1.1, 9.8]
stealth_ci   = [0.5, 0.5, 0.5, 1.3]
acc_mean2 = [99.4, 99.8, 99.8, 98.6]
acc_ci2   = [0.1, 0.1, 0.1, 0.2]

fig, ax1 = plt.subplots(figsize=(8.6, 5.2))
ax2 = ax1.twinx()

xs = np.arange(len(steps))

l1 = ax1.plot(xs, acc_mean2, marker="o", color=SLATE, linewidth=2.2, markersize=7,
              label="Accuracy (%) \u2014 left axis", zorder=4)
ax1.fill_between(xs, np.array(acc_mean2) - np.array(acc_ci2), np.array(acc_mean2) + np.array(acc_ci2),
                  color=SLATE, alpha=0.15, zorder=2)

l2 = ax2.plot(xs, stealth_mean, marker="s", color=RED, linewidth=2.2, markersize=7,
              label="Stealth-collusion ASR (%) \u2014 right axis", zorder=4)
ax2.fill_between(xs, np.array(stealth_mean) - np.array(stealth_ci), np.array(stealth_mean) + np.array(stealth_ci),
                  color=RED, alpha=0.15, zorder=2)

ax1.set_ylim(97.5, 100.3)
ax2.set_ylim(0, 12.5)
ax1.set_ylabel("Accuracy (%)", color=SLATE, fontsize=10.5, fontweight="bold")
ax2.set_ylabel("Stealth-Collusion ASR (%)", color=RED, fontsize=10.5, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=SLATE)
ax2.tick_params(axis="y", labelcolor=RED)
ax1.set_xticks(xs)
ax1.set_xticklabels(steps, fontsize=9.4)
ax1.grid(axis="y", color=LGREY, linewidth=0.8, zorder=0)
ax1.set_axisbelow(True)

for xi, v in zip(xs, acc_mean2):
    ax1.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points", xytext=(0, 10),
                 ha="center", fontsize=8.2, color=SLATE, fontweight="bold")
for xi, v in zip(xs, stealth_mean):
    ax2.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points", xytext=(0, -16),
                 ha="center", fontsize=8.2, color=RED, fontweight="bold")

ax1.axvspan(2.5, 3.5, color=GOLD, alpha=0.10, zorder=1)
ax1.text(3, 97.65, "cohesion-only routing never\nescalates stealth collusion\n(0.70 cohesion, attacked = clean) \u2014\nadding the contention signal\nexposes it, but single-query\nGWCC still has a real ceiling here.\nTrust store (excluded from this\nladder) is what closes it \u2014 see Table 1.",
         ha="center", fontsize=6.9, color="#6b5730", style="italic", linespacing=1.35)

ax1.set_title("Per-Ring Ablation Ladder \u2014 Dynamic Trust Store deliberately excluded\nmean \u00b1 95% CI, same 8 seeds as Table 1",
              fontsize=11.2, color=NAVY, fontweight="bold", pad=14)

lines = l1 + l2
labels = [ln.get_label() for ln in lines]
ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1,
           frameon=False, fontsize=8.8)

plt.tight_layout()
plt.savefig("/home/claude/report_build/figures/fig3_ablation_ladder.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print("saved fig3")
