import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT_PATH = HERE / "deimv2_dinov3_l_coco_map.png"

# --- Parsers ---
deim_pattern = re.compile(
    r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=   all \| maxDets=100 \] = ([0-9.]+)"
)
ult_pattern = re.compile(
    r"^\s+all\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s*$"
)

def parse_map(log_path, pattern):
    values = []
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                values.append(float(m.group(1)))
    return values

deim_map   = parse_map(HERE / "deimv2_dinov3_l_coco.log", deim_pattern)
repli_map  = parse_map(HERE / "deimv2_large_bs16.log", deim_pattern)
ult_map    = parse_map(HERE / "output.log", ult_pattern)

for label, vals in [("DEIM (original)", deim_map), ("DEIM (replication)", repli_map), ("Ultralytics", ult_map)]:
    best = max(vals)
    print(f"{label}: {len(vals)} epochs, best={best:.4f} @ ep{vals.index(best)}, final={vals[-1]:.4f}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 6))

COLORS = {"deim": "#1f77b4", "repli": "#2ca02c", "ult": "#ff7f0e"}

OFFSET = 0.003
ult_map_shifted = [v + OFFSET for v in ult_map]

series = [
    (deim_map,        COLORS["deim"],  "-",  "Original DEIMv2 repo (deimv2_dinov3_l_coco.log)"),
    (repli_map,       COLORS["repli"], "-",  "DEIMv2 repo replication (deimv2_large_bs16.log)"),
    (ult_map,         COLORS["ult"],   "-",  "Ultralytics training (output.log)"),
    (ult_map_shifted, COLORS["ult"],   "--", f"Ultralytics training +{OFFSET:.3f} offset (output.log)"),
]

for vals, color, ls, label in series:
    epochs = list(range(len(vals)))
    best_ep = vals.index(max(vals))
    best_v  = max(vals)

    ax.plot(epochs, vals, color=color, linewidth=2, linestyle=ls, label=label,
            alpha=0.6 if ls == "--" else 1.0)
    ax.scatter([best_ep], [best_v], color=color, s=60, zorder=5,
               marker="^" if ls == "--" else "o")
    if ls != "--":
        ax.annotate(
            f"Best: {best_v:.3f} @ ep{best_ep}",
            xy=(best_ep, best_v),
            xytext=(best_ep + 1.5, best_v + 0.0035),
            fontsize=9,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("mAP (COCO, IoU 0.50:0.95)", fontsize=12)
ax.set_title("DEIMv2 DINOv3-L COCO Training — mAP per Epoch", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.3f}"))
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"Saved → {OUT_PATH}")
