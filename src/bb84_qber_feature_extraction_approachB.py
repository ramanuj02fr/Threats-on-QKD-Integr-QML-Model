"""
Feature extraction script for Approach B (stealthy bounded perturbation).

For each poison_label (0,10,20,30,40,50) and each batch_id (0-99), treat
the 100 qber values in that batch as a small distribution and extract
13 features:
  - 4 statistical: mean_qber, std_qber, skew_qber, kurtosis_qber
    (skew/kurtosis computed manually with numpy, no scipy)
  - 9 bin-fraction: frac_qber_0.000 ... frac_qber_1.000
    (fraction of keys in that batch at each discrete qber value,
    since key_length=8 gives 9 possible discrete qber levels)

Input : poisoned_datasets/bb84_qber_classification_dataset_approachB.csv
Output: poisoned_datasets/qber_distribution_features_approachB.csv
        (600 rows = 6 poison levels x 100 batches, 15 columns =
         batch_id + poison_label + 13 features)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------- Config ----------------
INPUT_CSV = Path("poisoned_datasets/bb84_qber_classification_dataset_approachB.csv")
OUTPUT_CSV = Path("poisoned_datasets/qber_distribution_features_approachB.csv")

KEY_LENGTH = 8
# 9 discrete qber values possible for key_length=8: 0/8, 1/8, ..., 8/8
QBER_BINS = [round(i / KEY_LENGTH, 3) for i in range(KEY_LENGTH + 1)]


def manual_skew(x: np.ndarray) -> float:
    """Population (Fisher-Pearson) skewness, computed manually with numpy."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)  # population std (ddof=0)
    if std == 0:
        return 0.0
    return float(np.mean((x - mean) ** 3) / std ** 3)


def manual_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher's definition, normal = 0), computed manually."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)  # population std (ddof=0)
    if std == 0:
        return 0.0
    return float(np.mean((x - mean) ** 4) / std ** 4 - 3.0)


def extract_batch_features(qber_values: np.ndarray) -> dict:
    """Compute the 13 distribution features for one batch's qber values."""
    features = {
        "mean_qber": float(np.mean(qber_values)),
        "std_qber": float(np.std(qber_values)),
        "skew_qber": manual_skew(qber_values),
        "kurtosis_qber": manual_kurtosis(qber_values),
    }

    n = len(qber_values)
    for bin_val in QBER_BINS:
        col_name = f"frac_qber_{bin_val:.3f}"
        # np.isclose handles float rounding issues (e.g. 0.375 vs 0.3750000001)
        count = np.sum(np.isclose(qber_values, bin_val))
        features[col_name] = float(count / n)

    return features


def main():
    df = pd.read_csv(INPUT_CSV)

    rows = []
    for poison_label, group_by_label in df.groupby("poison_label"):
        for batch_id, group_by_batch in group_by_label.groupby("batch_id"):
            qber_values = group_by_batch["qber"].to_numpy()
            feats = extract_batch_features(qber_values)
            feats["batch_id"] = batch_id
            feats["poison_label"] = poison_label
            rows.append(feats)

    features_df = pd.DataFrame(rows)

    # reorder columns: batch_id, poison_label first, then the 13 features
    ordered_cols = ["batch_id", "poison_label", "mean_qber", "std_qber",
                     "skew_qber", "kurtosis_qber"] + \
                    [f"frac_qber_{b:.3f}" for b in QBER_BINS]
    features_df = features_df[ordered_cols]

    features_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Feature file -> {OUTPUT_CSV}")
    print(f"Shape: {features_df.shape}  (expected: (600, 15))")
    print(features_df.groupby("poison_label").size())
    print("\nSample rows:")
    print(features_df.head())


if __name__ == "__main__":
    main()