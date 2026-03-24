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

deim_map         = parse_map(HERE / "deimv2_dinov3_l_coco.log", deim_pattern)
repli_map        = parse_map(HERE / "deimv2_large_bs16.log", deim_pattern)
ult_bs16_map     = parse_map(HERE / "output_bs16.log", ult_pattern)
ult_cocoeval_map = parse_map(HERE / "output_coco_eval.log", deim_pattern)

for label, vals in [
    ("DEIM (original)",              deim_map),
    ("DEIM (replication bs16)",      repli_map),
    ("Ultralytics bs16",             ult_bs16_map),
    ("Ultralytics COCO eval",        ult_cocoeval_map),
]:
    best = max(vals)
    print(f"{label}: {len(vals)} epochs, best={best:.4f} @ ep{vals.index(best)}, final={vals[-1]:.4f}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 6))

OFFSET = 0.003
ult_bs16_map_shifted = [v + OFFSET for v in ult_bs16_map]

COLORS = {"deim": "#1f77b4", "repli": "#2ca02c", "ult": "#ff7f0e", "ult_bs16": "#d62728", "ult_cocoeval": "#9467bd"}

# Toggle any series by commenting out its line
series = [
    (deim_map,             COLORS["deim"],        "Original DEIMv2 repo (deimv2_dinov3_l_coco.log)"),
    (repli_map,            COLORS["repli"],       "DEIMv2 repo replication bs16 (deimv2_large_bs16.log)"),
    (ult_cocoeval_map,     COLORS["ult_cocoeval"],"Ultralytics COCO eval (output_coco_eval.log)"),
    # (ult_bs16_map_shifted, COLORS["ult_bs16"],    "Ultralytics training bs16 (output_bs16.log)"),
]

for vals, color, label in series:
    epochs = list(range(len(vals)))
    ax.plot(epochs, vals, color=color, linewidth=2, linestyle="-", label=label)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("mAP (COCO, IoU 0.50:0.95)", fontsize=12)
ax.set_title("DEIMv2 DINOv3-L COCO Training — mAP per Epoch", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.3f}"))
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"Saved → {OUT_PATH}")
