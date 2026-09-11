"""
Approach B: Stealthy Bounded Perturbation poisoning script.

Mechanism: Same random p% row selection as Approach A, but instead of
fully resampling mismatched_bits, we only apply a small bounded shift
(+1 or -1 bit, random direction, clipped to [0, key_length]). This
keeps the poisoning "stealthy" -- close to the original value instead
of a full random replacement.

Input : src/bb84_qber_dataset.csv   (clean dataset, same as Approach A)
Output: poisoned_datasets/bb84_qber_dataset_approachB_{p}pct.csv   (p in 0,10,20,30,40,50)
        poisoned_datasets/bb84_qber_classification_dataset_approachB.csv  (combined, all levels)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------- Config ----------------
INPUT_CSV = Path("bb84_qber_dataset.csv")
OUTPUT_DIR = Path("poisoned_datasets")
POISON_LEVELS = [0, 10, 20, 30, 40, 50]   # percentage of rows to poison per level
SHIFT_MAGNITUDE = 1                        # bounded shift size in bits (+/-1)
RANDOM_SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def bounded_shift_row(mismatched_bits: int, key_length: int, rng: np.random.Generator) -> int:
    """
    Apply a random +1 or -1 bit shift to mismatched_bits, clipped to [0, key_length].

    Edge cases:
      - if mismatched_bits == 0, only +1 is possible (can't go below 0)
      - if mismatched_bits == key_length, only -1 is possible (can't exceed key_length)
    """
    if mismatched_bits == 0:
        delta = SHIFT_MAGNITUDE
    elif mismatched_bits == key_length:
        delta = -SHIFT_MAGNITUDE
    else:
        delta = rng.choice([-SHIFT_MAGNITUDE, SHIFT_MAGNITUDE])
    new_val = mismatched_bits + delta
    return int(np.clip(new_val, 0, key_length))


def poison_dataset(clean_df: pd.DataFrame, poison_pct: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Take a fresh copy of the clean dataset, randomly select poison_pct% of
    rows (without replacement), apply a bounded +/-1 shift to
    mismatched_bits for those rows, recompute qber, and tag every row
    with poison_label + was_corrupted.
    """
    df = clean_df.copy()
    n_rows = len(df)
    n_poison = int(round(n_rows * poison_pct / 100))

    df["was_corrupted"] = 0
    df["poison_label"] = poison_pct

    if n_poison > 0:
        poison_idx = rng.choice(df.index, size=n_poison, replace=False)

        for idx in poison_idx:
            key_length = df.at[idx, "key_length"]
            old_mismatched = df.at[idx, "mismatched_bits"]
            new_mismatched = bounded_shift_row(old_mismatched, key_length, rng)
            df.at[idx, "mismatched_bits"] = new_mismatched
            df.at[idx, "qber"] = new_mismatched / key_length
            df.at[idx, "was_corrupted"] = 1

    return df


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    clean_df = pd.read_csv(INPUT_CSV)

    all_levels = []
    for pct in POISON_LEVELS:
        poisoned_df = poison_dataset(clean_df, pct, rng)
        out_path = OUTPUT_DIR / f"bb84_qber_dataset_approachB_{pct}pct.csv"
        poisoned_df.to_csv(out_path, index=False)
        print(f"[OK] poison_label={pct:>2}%  ->  {out_path}  "
              f"(rows={len(poisoned_df)}, corrupted={poisoned_df['was_corrupted'].sum()})")
        all_levels.append(poisoned_df)

    combined_df = pd.concat(all_levels, ignore_index=True)
    combined_path = OUTPUT_DIR / "bb84_qber_classification_dataset_approachB.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\n[OK] Combined dataset -> {combined_path}  (rows={len(combined_df)})")
    print(combined_df["poison_label"].value_counts().sort_index())


if __name__ == "__main__":
    main()