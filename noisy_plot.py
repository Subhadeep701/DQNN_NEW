import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ===== Styling & Config =====
mpl.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
})

# ===== Load CSV =====
filename = "3_4_3_entropy_noisy_results.csv"
df = pd.read_csv(filename)

# Unique metrics in the file
metrics = df["Metric"].unique()

# ===== Figure =====
plt.figure(figsize=(7, 6))

# Define colors and markers
colors = {
    "train_entropy": "#1f77b4",     # blue
    "test_entropy": "#ff7f0e",      # orange
    "noisy_entropy_0.1": "#2ca02c", # green
    "noisy_entropy_0.2": "#d62728", # red
    "noisy_entropy_0.3": "#9467bd"  # purple
}

markers = {
    "train_entropy": "o",
    "test_entropy": "s",
    "noisy_entropy_0.1": "D",
    "noisy_entropy_0.2": "v",
    "noisy_entropy_0.3": "^"
}

# ===== Plot each metric =====
for metric in metrics:
    subset = df[df["Metric"] == metric]

    # Legend label formatting
    if "noisy_entropy" in metric:
        nl = metric.split("_")[-1]
        label = fr"Noisy Test ($\zeta = {nl}$)"
    elif metric == "train_entropy":
        label = "Train"
    elif metric == "test_entropy":
        label = "Test"
    else:
        label = metric.replace("_", " ").capitalize()

    plt.errorbar(
        subset["TrainingRound"],
        subset["Mean"],
        yerr=subset["CI"],
        fmt='-{}'.format(markers.get(metric, 'o')),  # solid line + marker
        color=colors.get(metric, "gray"),
        capsize=4,
        elinewidth=1.2,
        alpha=0.9,
        label=label
    )

# ===== Labels & Styling =====
plt.xlabel("Training Rounds")
plt.ylabel("Entropy Cost")
# plt.title("QNN Entropy Performance", fontweight="bold")

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=True, fancybox=True, shadow=False, loc="best")
plt.tight_layout()

# ===== Save & Show =====
plt.savefig("3_4_3_QNN_Entropy_From_CSV.png", dpi=300, bbox_inches="tight")
plt.show()
