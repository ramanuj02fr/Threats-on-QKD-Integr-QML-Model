"""
bb84_qber_poisoning.py

Generates poisoned versions of the clean BB84 QBER dataset (Dataset 1)
using Approach A: discrete resampling of QBER values.

For each poisoning level p in {0, 10, 20, 30, 40, 50} (percent):
    - Take a fresh copy of the clean dataset (all rows).
    - Randomly select p% of the rows (without replacement).
    - For each selected row, resample `mismatched_bits` uniformly from
      0..key_length (so the corrupted QBER stays within the same set of
      discrete values a real key of that length can produce), and
      recompute qber = mismatched_bits / key_length.
    - Leave the remaining (100-p)% rows untouched.
    - Label the WHOLE resulting dataset (all rows) with poison_label = p.

Each poisoned version is saved as its own CSV (named by its poison
percentage), and all versions are also concatenated into one combined
classification dataset with a `poison_label` column as the target class.

Usage:
    python bb84_qber_poisoning.py
    python bb84_qber_poisoning.py --input bb84_qber_dataset.csv --levels 0 10 20 30 40 50 --seed 42 --out-dir poisoned_datasets
"""

import argparse
import os

import numpy as np
import pandas as pd


def poison_dataset(df: pd.DataFrame, poison_pct: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Return a poisoned copy of `df` where `poison_pct`% of rows have their
    mismatched_bits/qber resampled uniformly at random (Approach A).
    The whole returned dataframe is tagged with poison_label = poison_pct.
    """
    poisoned = df.copy()
    n_rows = len(poisoned)
    n_poison = int(round(n_rows * poison_pct / 100))

    if n_poison > 0:
        poison_idx = rng.choice(n_rows, size=n_poison, replace=False)

        key_length = poisoned["key_length"].to_numpy()
        new_mismatched = rng.integers(0, key_length[poison_idx] + 1)  # inclusive of key_length

        poisoned.loc[poisoned.index[poison_idx], "mismatched_bits"] = new_mismatched
        poisoned.loc[poisoned.index[poison_idx], "qber"] = (
            new_mismatched / key_length[poison_idx]
        )
        poisoned["was_corrupted"] = 0
        poisoned.loc[poisoned.index[poison_idx], "was_corrupted"] = 1
    else:
        poisoned["was_corrupted"] = 0

    poisoned["poison_label"] = poison_pct
    return poisoned


def generate_all_versions(
    input_csv: str,
    levels: list[int],
    seed: int,
    out_dir: str,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    rng = np.random.default_rng(seed)

    os.makedirs(out_dir, exist_ok=True)

    all_versions = []
    for p in levels:
        poisoned_df = poison_dataset(df, p, rng)
        filename = os.path.join(out_dir, f"bb84_qber_dataset_poison_{p}pct.csv")
        poisoned_df.to_csv(filename, index=False)

        n_corrupted = int(poisoned_df["was_corrupted"].sum())
        print(f"[{p:>3}% poisoning] saved -> {filename} "
              f"({n_corrupted}/{len(poisoned_df)} rows corrupted, "
              f"mean QBER = {poisoned_df['qber'].mean():.4f})")

        all_versions.append(poisoned_df)

    combined = pd.concat(all_versions, ignore_index=True)
    combined_path = os.path.join(out_dir, "bb84_qber_classification_dataset.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined classification dataset -> {combined_path} "
          f"({len(combined)} rows, {combined['poison_label'].nunique()} classes)")

    return combined


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate poisoned BB84 QBER dataset versions (Approach A: discrete resampling)."
    )
    parser.add_argument("--input", type=str, default="bb84_qber_dataset.csv",
                         help="Path to the clean Dataset 1 CSV (default: bb84_qber_dataset.csv)")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 10, 20, 30, 40, 50],
                         help="Poisoning percentages to generate (default: 0 10 20 30 40 50)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--out-dir", type=str, default="poisoned_datasets",
                         help="Output directory for generated CSVs (default: poisoned_datasets)")
    return parser.parse_args()


def main():
    args = parse_args()
    generate_all_versions(
        input_csv=args.input,
        levels=args.levels,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()