"""
bb84_qber_feature_extraction.py

Builds the final classification-ready feature table from the poisoned
BB84 QBER datasets (Approach A, see bb84_qber_poisoning.py).

For each poison level (0, 10, 20, 30, 40, 50 %) and each batch (0-99)
within that level, this script treats the 100 QBER values in that batch
as a small empirical distribution and extracts:

  - 4 statistical parameters: mean, std, skewness, kurtosis
  - 9 bin-fraction features: the proportion of keys in that batch whose
    QBER equals each of the 9 possible discrete values a key of
    key_length=8 can produce (0/8, 1/8, ..., 8/8)

That gives 13 features per batch. With 100 batches x 6 poison levels,
the final table has 600 rows, each labeled with `poison_label`
(the classification target).

The script also saves an overlaid histogram of QBER values per poison
level (aggregated across all batches of that level) for visual
inspection of how the distribution shifts as poisoning increases.

Usage:
    python bb84_qber_feature_extraction.py
    python bb84_qber_feature_extraction.py --in-dir poisoned_datasets --levels 0 10 20 30 40 50 --out features.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _skew(x: np.ndarray) -> float:
    """Fisher-Pearson skewness (population), matches scipy.stats.skew default."""
    x = np.asarray(x, dtype=float)
    std = x.std()
    if std == 0:
        return 0.0
    return np.mean((x - x.mean()) ** 3) / std ** 3


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (population, Fisher's definition), matches scipy.stats.kurtosis default."""
    x = np.asarray(x, dtype=float)
    std = x.std()
    if std == 0:
        return 0.0
    return np.mean((x - x.mean()) ** 4) / std ** 4 - 3.0


def extract_batch_features(qber_values: np.ndarray, key_length: int) -> dict:
    """
    Given the QBER values of a single batch (100 keys), compute the
    4 statistical parameters + 9 bin-fraction features.
    """
    n = len(qber_values)
    features = {
        "mean_qber": np.mean(qber_values),
        "std_qber": np.std(qber_values),
        "skew_qber": _skew(qber_values),
        "kurtosis_qber": _kurtosis(qber_values),
    }

    # The 9 possible discrete QBER values for this key_length: 0/L, 1/L, ..., L/L
    possible_values = np.arange(key_length + 1) / key_length
    for v in possible_values:
        # Use isclose to avoid floating point equality issues
        frac = np.mean(np.isclose(qber_values, v))
        features[f"frac_qber_{v:.3f}"] = frac

    return features


def build_feature_table(in_dir: str, levels: list[int]) -> pd.DataFrame:
    rows = []

    for p in levels:
        path = os.path.join(in_dir, f"bb84_qber_dataset_poison_{p}pct.csv")
        df = pd.read_csv(path)
        key_length = int(df["key_length"].iloc[0])

        for batch_id, batch_df in df.groupby("batch_id"):
            feats = extract_batch_features(batch_df["qber"].to_numpy(), key_length)
            feats["batch_id"] = batch_id
            feats["poison_label"] = p
            rows.append(feats)

    feature_df = pd.DataFrame(rows)

    # Reorder columns: batch_id, poison_label first, then features
    front_cols = ["batch_id", "poison_label"]
    other_cols = [c for c in feature_df.columns if c not in front_cols]
    feature_df = feature_df[front_cols + other_cols]

    return feature_df


def plot_histograms(in_dir: str, levels: list[int], out_path: str):
    """
    Small-multiples histogram: one subplot per poison level (same y-axis
    scale across all subplots), plus a combined line-plot overlay panel,
    so the distribution shift is clearly visible without color blending.
    """
    n_levels = len(levels)
    fig, axes = plt.subplots(2, n_levels, figsize=(3 * n_levels, 7),
                              gridspec_kw={"height_ratios": [1, 1]})

    colors = plt.cm.viridis(np.linspace(0, 0.9, n_levels))
    level_data = {}
    key_length = None

    # Row 1: one bar-histogram subplot per poison level (shared y-axis)
    for i, p in enumerate(levels):
        path = os.path.join(in_dir, f"bb84_qber_dataset_poison_{p}pct.csv")
        df = pd.read_csv(path)
        key_length = int(df["key_length"].iloc[0])
        level_data[p] = df["qber"].to_numpy()

    bins = np.arange(key_length + 2) / key_length - (0.5 / key_length)
    bin_centers = np.arange(key_length + 1) / key_length
    max_density = 0
    hist_counts = {}
    for p in levels:
        counts, _ = np.histogram(level_data[p], bins=bins, density=True)
        hist_counts[p] = counts
        max_density = max(max_density, counts.max())

    for i, p in enumerate(levels):
        ax = axes[0, i]
        ax.bar(bin_centers, hist_counts[p], width=1 / key_length * 0.85, color=colors[i])
        ax.set_title(f"{p}% poisoned", fontsize=10)
        ax.set_ylim(0, max_density * 1.1)
        ax.set_xlabel("QBER", fontsize=8)
        if i == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)

    # Row 2 (merged into one wide panel): line-plot overlay for direct comparison
    for ax in axes[1, :]:
        ax.remove()
    ax_overlay = fig.add_subplot(2, 1, 2)
    for i, p in enumerate(levels):
        ax_overlay.plot(bin_centers, hist_counts[p], marker="o", color=colors[i],
                         label=f"{p}% poisoned", linewidth=2)
    ax_overlay.set_xlabel("QBER")
    ax_overlay.set_ylabel("Density")
    ax_overlay.set_title("Overlay: QBER Distribution Shift Across Poisoning Levels")
    ax_overlay.legend(ncol=len(levels), fontsize=8, loc="upper center")

    fig.suptitle("QBER Distribution by Poisoning Level", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Histogram saved -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract per-batch distribution features from poisoned BB84 QBER datasets."
    )
    parser.add_argument("--in-dir", type=str, default="poisoned_datasets",
                         help="Directory containing the poisoned dataset CSVs (default: poisoned_datasets)")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 10, 20, 30, 40, 50],
                         help="Poisoning percentages to include (default: 0 10 20 30 40 50)")
    parser.add_argument("--out", type=str, default="qber_distribution_features.csv",
                         help="Output CSV path for the feature table")
    parser.add_argument("--hist-out", type=str, default="qber_distribution_histogram.png",
                         help="Output path for the histogram image")
    return parser.parse_args()


def main():
    args = parse_args()

    feature_df = build_feature_table(args.in_dir, args.levels)
    feature_df.to_csv(args.out, index=False)
    print(f"Feature table saved -> {args.out} ({len(feature_df)} rows, {feature_df.shape[1]} columns)")

    print("\nMean feature values per poison level:")
    print(feature_df.groupby("poison_label")[["mean_qber", "std_qber", "skew_qber", "kurtosis_qber"]].mean())

    plot_histograms(args.in_dir, args.levels, args.hist_out)


if __name__ == "__main__":
    main()