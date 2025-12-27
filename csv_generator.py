import pandas as pd
from pathlib import Path

# --- user-editable list of compact architectures ---
arch_compact = ["111","131","12321","222","242","2332","23432","333","343","3443"]

# base folder where *_results.csv are stored; change if necessary
base_path = Path(".")

out_rows = []

for compact in arch_compact:
    # convert compact form '12321' -> '1_2_3_2_1'
    arch_str = "_".join(list(compact))
    filename = base_path / f"{arch_str}_results.csv"

    if not filename.exists():
        print(f"[WARN] File not found: {filename}  (skipping {arch_str})")
        continue

    df = pd.read_csv(filename)

    # ensure TrainingRound is present and numeric
    if "TrainingRound" not in df.columns:
        print(f"[WARN] 'TrainingRound' column missing in {filename}; skipping")
        continue

    # Keep only the entropy rows and the two methods
    filtered = df[
        (df["Metric"].isin(["train_entropy", "test_entropy"])) &
        (df["Method"].isin(["Entropy", "Fidelity"]))
    ].copy()

    if filtered.empty:
        print(f"[WARN] No train/test entropy rows in {filename}; skipping")
        continue

    # sort by TrainingRound and take last entry per (Method, Metric)
    filtered = filtered.sort_values("TrainingRound")
    last_rows = filtered.groupby(["Method", "Metric"], as_index=False).last()

    # ensure we always emit the four expected combos (Entropy/Fidelity x train/test)
    for method_label in ["Entropy", "Fidelity"]:
        for metric in ["train_entropy", "test_entropy"]:
            match = last_rows[
                (last_rows["Method"] == method_label) &
                (last_rows["Metric"] == metric)
            ]
            if not match.empty:
                r = match.iloc[0]
                out_rows.append({
                    "Architecture": arch_str,
                    "Method": method_label,
                    "Metric": metric,
                    "TrainingRound": int(r["TrainingRound"]),
                    "Mean": r.get("Mean", pd.NA),
                    "CI": r.get("CI", pd.NA)
                })
            else:
                # missing combination -> write NaNs so CSV has consistent rows
                out_rows.append({
                    "Architecture": arch_str,
                    "Method": method_label,
                    "Metric": metric,
                    "TrainingRound": pd.NA,
                    "Mean": pd.NA,
                    "CI": pd.NA
                })

# build combined DataFrame and save
combined_df = pd.DataFrame(out_rows)

out_file = base_path / "all_arch_entropy_last_round_summary.csv"
combined_df.to_csv(out_file, index=False)
print(f"\n✅ Saved combined summary to: {out_file}")
print(f"Rows written: {len(combined_df)}")
