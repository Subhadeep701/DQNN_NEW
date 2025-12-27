import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Architectures and CSVs
# -----------------------------
architectures = ["2_2_2", "2_4_2", "2_3_3_2", "2_3_4_3_2"]  # list of architectures
csv_files = [f"{arch}_results.csv" for arch in architectures]

# -----------------------------
# Plotting function for multi-architecture + both methods
# -----------------------------
def plot_metric_multi_arch(csv_files, architectures, metric_key, ylabel, filename):
    plt.figure(figsize=(7, 5.5))

    colors = ["blue", "cyan", "green", "lime", "orange", "red", "purple", "magenta"]
    styles = {"A": "-", "B": "--"}
    markers = {"A": "o", "B": "s"}

    color_idx = 0
    for csv_file, arch in zip(csv_files, architectures):
        df = pd.read_csv(csv_file)

        # ✅ Filter to only include rounds up to 150
        df = df[df['TrainingRound'] <= 150]

        for method in ["A", "B"]:
            method_label = "Entropy" if method == "A" else "Fidelity"
            df_subset = df[(df['Metric'] == metric_key) & (df['Method'] == method_label)].sort_values('TrainingRound')

            x_axis = df_subset['TrainingRound'].values
            mean_vals = df_subset['Mean'].values
            ci_vals = df_subset['CI'].values

            label = f"{arch} - Proposed (Relative Entropy based)" if method == "A" else f"{arch} - Existing (Fidelity based)"
            plt.errorbar(
                x_axis, mean_vals, yerr=ci_vals,
                fmt=markers[method] + styles[method],
                color=colors[color_idx],
                capsize=4,
                label=label
            )
            color_idx += 1

    plt.xlabel("Training Rounds", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# -----------------------------
# Generate plots for all metrics
# -----------------------------
metrics = ["train_entropy", "cv_entropy", "test_entropy", "train_fidelity", "cv_fidelity", "test_fidelity"]
ylabels = ["Entropy", "Entropy", "Entropy", "Fidelity", "Fidelity", "Fidelity"]

for metric, ylabel in zip(metrics, ylabels):
    plot_metric_multi_arch(csv_files, architectures, metric,
                           ylabel, f"multi_arch_{metric}_comparison.png")
