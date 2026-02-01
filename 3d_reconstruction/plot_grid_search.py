import matplotlib.pyplot as plt
import numpy as np

results = [
    {"score": 0.069706, "smooth": "z_gaussian",  "low": 0.35, "high": 0.65, "conn": 18},
    {"score": 0.069706, "smooth": "z_gaussian",  "low": 0.35, "high": 0.65, "conn": 26},
    {"score": 0.072276, "smooth": "gaussian_3d", "low": 0.35, "high": 0.65, "conn": 18},
    {"score": 0.072276, "smooth": "gaussian_3d", "low": 0.35, "high": 0.65, "conn": 26},
    {"score": 0.172776, "smooth": "gaussian_3d", "low": 0.25, "high": 0.55, "conn": 18},
    {"score": 0.172776, "smooth": "gaussian_3d", "low": 0.25, "high": 0.55, "conn": 26},
    {"score": 0.193516, "smooth": "z_gaussian",  "low": 0.25, "high": 0.55, "conn": 18},
    {"score": 0.193516, "smooth": "z_gaussian",  "low": 0.25, "high": 0.55, "conn": 26},
]

# -----------------------------
# Plot 1: Dot plot (best single chart)
# -----------------------------
smooth_order = ["gaussian_3d", "z_gaussian"]
x_map = {name: i for i, name in enumerate(smooth_order)}

# jitter to separate points with same x
rng = np.random.default_rng(0)

plt.figure(figsize=(9, 5))

for r in results:
    x = x_map[r["smooth"]]
    # small jitter so conn 18/26 don't overlap perfectly
    jitter = (rng.random() - 0.5) * 0.10
    xj = x + jitter

    marker = "o" if r["conn"] == 18 else "s"  # 18: circle, 26: square
    plt.scatter(xj, r["score"], marker=marker)

    # annotate with (low,high)
    lbl = f"({r['low']:.2f},{r['high']:.2f})"
    plt.annotate(lbl, (xj, r["score"]), textcoords="offset points", xytext=(6, 4), fontsize=9)

plt.xticks([0, 1], smooth_order)
plt.xlabel("Smoothing method")
plt.ylabel("Mean |relative volume diff| (lower is better)")
plt.title("Grid search results: effect of smoothing / hysteresis thresholds / connectivity")

# custom legend
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', linestyle='None', label='connectivity=18'),
    Line2D([0], [0], marker='s', linestyle='None', label='connectivity=26'),
]
plt.legend(handles=legend_elems, loc="best")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# -----------------------------
# Plot 2 (optional): Heatmaps (2 panels) showing thresholds vs connectivity
# -----------------------------
# Build a small grid: rows = threshold pairs, cols = connectivity
pairs = [(0.25, 0.55), (0.35, 0.65)]
conns = [18, 26]

def score_lookup(smooth, low, high, conn):
    for r in results:
        if r["smooth"] == smooth and r["low"] == low and r["high"] == high and r["conn"] == conn:
            return r["score"]
    return np.nan

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

for ax, smooth in zip(axes, smooth_order):
    mat = np.array([[score_lookup(smooth, low, high, conn) for conn in conns] for (low, high) in pairs], dtype=float)

    im = ax.imshow(mat, aspect="auto")  
    ax.set_title(smooth)
    ax.set_xticks(range(len(conns)))
    ax.set_xticklabels([str(c) for c in conns])
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"low={low:.2f}\nhigh={high:.2f}" for (low, high) in pairs])
    ax.set_xlabel("Connectivity")
    ax.set_ylabel("Hysteresis thresholds")

    # annotate values
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.6f}", ha="center", va="center", fontsize=9)

fig.colorbar(im, ax=axes, shrink=0.9, label="Mean |relative volume diff|")
plt.show()
