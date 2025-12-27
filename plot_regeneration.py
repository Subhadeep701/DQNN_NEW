import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load CSV
# -----------------------------
arch_str = "3_4_3"  # Replace with your architecture string
filename = f"{arch_str}_results.csv"
df = pd.read_csv(filename)

# Extract unique training rounds for x-axis
x_axis_values = np.sort(df['TrainingRound'].unique())

# -----------------------------
# Plotting function following original style
# -----------------------------
def plot_metric_csv_style(metric_key, ylabel, title, filename):
    plt.figure(figsize=(6,5))
    for method, style, color in [("A","o-","blue"), ("B","s--","red")]:
        # Select the correct Method from CSV
        method_label = "Entropy" if method=="A" else "Fidelity"
        df_subset = df[(df['Metric'] == metric_key) & (df['Method'] == method_label)]
        df_subset = df_subset.sort_values('TrainingRound')
        mean_vals = df_subset['Mean'].values
        ci_vals = df_subset['CI'].values
        label = "Proposed (Relative Entropy based)" if method=="A" else "Existing (Fidelity based)"
        plt.errorbar(x_axis_values, mean_vals, yerr=ci_vals, fmt=style,
                     color=color, capsize=4, label=label)
    plt.xlabel("Training Rounds")
    plt.ylabel(ylabel)
    #plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# -----------------------------
# Generate plots for all metrics
# -----------------------------
metrics = ["train_entropy","cv_entropy","test_entropy","train_fidelity","cv_fidelity","test_fidelity"]
ylabels = ["Entropy","Entropy","Entropy","Fidelity","Fidelity","Fidelity"]

for metric, ylabel in zip(metrics, ylabels):
    plot_metric_csv_style(metric, ylabel, f"{metric.replace('_',' ').upper()} Comparison",
                          f"{arch_str}_{metric}_comparison.png")
