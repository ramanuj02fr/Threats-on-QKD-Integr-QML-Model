"""
bb84_qber_comparison_plot_AB.py

Visual comparison of Approach A (random resample) vs Approach B
(stealthy bounded +/-1 shift) poisoning, showing how the QBER
distribution shifts as poisoning increases under each mechanism.

Follows the same plotting style as the original
bb84_qber_feature_extraction.py::plot_histograms() (viridis colormap,
shared y-axis bar histograms, overlay line-plot), extended to a
3-row layout so Approach A and Approach B can be compared directly:

  Row 1: Approach A histograms, one subplot per poison level (shared y-axis)
  Row 2: Approach B histograms, one subplot per poison level (SAME y-axis
         scale as row 1, so the visual "flatness" of B vs A is honest)
  Row 3: Overlay line-plot with both approaches on the same axes
         (solid lines = Approach A, dashed lines = Approach B),
         same viridis color per poison level in both.

Input : poisoned_datasets/bb84_qber_dataset_poison_{p}pct.csv       (Approach A)
        poisoned_datasets/bb84_qber_dataset_approachB_{p}pct.csv    (Approach B)
Output: poisoned_datasets/qber_comparison_plot_AB.png
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_qber_values(in_dir: str, levels: list[int], filename_template: str) -> tuple[dict, int]:
    """Load QBER values per poison level from CSVs matching filename_template."""
    level_data = {}
    key_length = None
    for p in levels:
        path = os.path.join(in_dir, filename_template.format(p=p))
        df = pd.read_csv(path)
        key_length = int(df["key_length"].iloc[0])
        level_data[p] = df["qber"].to_numpy()
    return level_data, key_length


def compute_histograms(level_data: dict, levels: list[int], key_length: int):
    """Compute normalized density histograms over the 9 discrete QBER bins."""
    bins = np.arange(key_length + 2) / key_length - (0.5 / key_length)
    bin_centers = np.arange(key_length + 1) / key_length

    hist_counts = {}
    for p in levels:
        counts, _ = np.histogram(level_data[p], bins=bins, density=True)
        hist_counts[p] = counts

    return bin_centers, hist_counts


def plot_comparison(in_dir: str, levels: list[int], out_path: str):
    n_levels = len(levels)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_levels))

    # --- Load both approaches ---
    data_a, key_length = load_qber_values(in_dir, levels, "bb84_qber_dataset_poison_{p}pct.csv")
    data_b, _ = load_qber_values(in_dir, levels, "bb84_qber_dataset_approachB_{p}pct.csv")

    bin_centers, hist_a = compute_histograms(data_a, levels, key_length)
    _, hist_b = compute_histograms(data_b, levels, key_length)

    # Shared y-axis scale across BOTH rows, so the flatness of Approach B
    # relative to Approach A is visually honest (not auto-rescaled per row).
    max_density = max(
        max(counts.max() for counts in hist_a.values()),
        max(counts.max() for counts in hist_b.values()),
    )

    fig = plt.figure(figsize=(3 * n_levels, 11))
    gs = fig.add_gridspec(3, n_levels, height_ratios=[1, 1, 1.3])

    # Row 1: Approach A histograms
    for i, p in enumerate(levels):
        ax = fig.add_subplot(gs[0, i])
        ax.bar(bin_centers, hist_a[p], width=1 / key_length * 0.85, color=colors[i])
        ax.set_title(f"{p}% poisoned", fontsize=10)
        ax.set_ylim(0, max_density * 1.1)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_ylabel("Density\n(Approach A)", fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_xticklabels([])

    # Row 2: Approach B histograms (same y-scale as row 1)
    for i, p in enumerate(levels):
        ax = fig.add_subplot(gs[1, i])
        ax.bar(bin_centers, hist_b[p], width=1 / key_length * 0.85, color=colors[i])
        ax.set_ylim(0, max_density * 1.1)
        ax.set_xlabel("QBER", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_ylabel("Density\n(Approach B)", fontsize=9)
        else:
            ax.set_yticklabels([])

    # Row 3: overlay line-plot, both approaches on the same axes
    ax_overlay = fig.add_subplot(gs[2, :])
    for i, p in enumerate(levels):
        ax_overlay.plot(bin_centers, hist_a[p], marker="o", linestyle="-",
                         color=colors[i], linewidth=2, label=f"A: {p}%")
        ax_overlay.plot(bin_centers, hist_b[p], marker="s", linestyle="--",
                         color=colors[i], linewidth=2, label=f"B: {p}%")
    ax_overlay.set_xlabel("QBER")
    ax_overlay.set_ylabel("Density")
    ax_overlay.set_title("Overlay: Approach A (solid) vs Approach B (dashed)")
    ax_overlay.legend(ncol=n_levels, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15))

    fig.suptitle("QBER Distribution Shift: Approach A (random resample) vs Approach B (stealthy \u00b11-bit shift)",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare QBER distribution shift between Approach A and Approach B poisoning."
    )
    parser.add_argument("--in-dir", type=str, default="poisoned_datasets",
                         help="Directory containing both approaches' poisoned dataset CSVs")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 10, 20, 30, 40, 50],
                         help="Poisoning percentages to include")
    parser.add_argument("--out", type=str, default="poisoned_datasets/qber_comparison_plot_AB.png",
                         help="Output path for the comparison image")
    return parser.parse_args()


def main():
    args = parse_args()
    plot_comparison(args.in_dir, args.levels, args.out)


if __name__ == "__main__":
    main()