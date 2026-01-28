import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ================= Configuration =================
output_dir = "/home/raza/EMRV1/EMRV2/outputs/learning_curve_graphics"
output_file = "learning_curves_metrics.png"
save_figure = True  # Set False if you only want to display

# X-axis: number of samples
samples = [10, 20, 30, 50, 67]

# ================= Real results =================
model_a = {
    "R-1":  [51.36, 55.52, 58.88, 63.57, 67.19],
    "R-2":  [22.35, 25.68, 30.25, 37.95, 45.66],
    "R-L":  [30.52, 33.36, 37.19, 40.51, 46.21],
    "BS":  [83.80, 85.01, 87.41, 85.79, 90.84],
    "BLE": [35.08, 41.33, 37.98, 44.71, 46.73],
    "MET": [29.41, 34.01, 36.06, 42.81, 47.51],
}

model_b = {
    "R-1":  [24.29, 29.17, 23.64, 34.25, 35.67],
    "R-2":  [6.71, 8.32, 5.83, 9.36, 10.82],
    "R-L":  [12.87, 15.04, 11.33, 13.67, 14.66],
    "BS":  [81.20, 81.19, 77.56, 78.00, 79.58],
    "BLE": [38.07, 35.12, 28.05, 30.18, 30.58],
    "MET": [13.38, 16.92, 14.84, 25.94, 25.57],
}

# ================= Plot =================
fig, ax = plt.subplots(figsize=(10, 5))  # IMPORTANT: wider canvas
fig.subplots_adjust(right=0.75)          # reserve space for legends

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
metrics = list(model_a.keys())

for i, metric in enumerate(metrics):
    color = colors[i]
    ax.plot(samples, model_a[metric], linestyle='-', color=color)
    ax.plot(samples, model_b[metric], linestyle=':', color=color)

ax.set_xlabel("Number of Training Samples")
ax.set_ylabel("Score")

# ================= Legends =================
# Model legend (line style)
model_legend = [
    Line2D([0], [0], color='black', linestyle='-', label='ClinGen-DSDA'),
    Line2D([0], [0], color='black', linestyle=':', label='Vanilla LLaMA'),
]

# Metric legend (color)
metric_legend = [
    Line2D([0], [0], color=colors[i], linestyle='-', label=metric)
    for i, metric in enumerate(metrics)
]

legend1 = ax.legend(
    handles=model_legend,
    # title="Model",
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    frameon=True
)

legend2 = ax.legend(
    handles=metric_legend,
    title="Metrics",
    loc="lower left",
    bbox_to_anchor=(1.01, 0.0),
    frameon=True
)

ax.add_artist(legend1)

ax.grid(True)

# ================= Save =================
if save_figure:
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, output_file)
    plt.savefig(save_path, dpi=300)

plt.show()
