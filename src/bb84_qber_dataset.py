"""
bb84_qber_dataset.py

Generates a BB84 QBER dataset with Eve present (Intercept-Resend attack).

Plan:
1. Fix key length (default 8 bits). Only matched bases are assumed to
   exist (i.e. we directly simulate the sifted bits — no basis-mismatch
   discard step needed).
2. Alice and Bob attempt to generate `keys_per_batch` keys (default 100),
   each `key_length` bits long.
3. Every key generation goes through an Intercept-Resend (IR) attack
   simulation by Eve, and the resulting QBER for that key is recorded.
   Not every key will show a QBER > 0 — that's expected.
4. Step 3 is repeated `num_batches` times (default 100), producing a
   distribution of QBER values rather than a single number.

Usage:
    python bb84_qber_dataset.py
    python bb84_qber_dataset.py --key-length 8 --keys-per-batch 100 --num-batches 100 --seed 42 --out bb84_qber_dataset.csv
"""

import argparse
import numpy as np
import pandas as pd


def generate_key_with_ir_attack(key_length: int, rng: np.random.Generator):
    """
    Simulate a single BB84 key exchange (sifted / matched-bases-only)
    under an Intercept-Resend attack by Eve.
    """
    alice_bits = rng.integers(0, 2, size=key_length)
    alice_bases = rng.integers(0, 2, size=key_length)  # 0 = '+', 1 = 'x'

    eve_bases = rng.integers(0, 2, size=key_length)
    eve_bits = np.where(
        eve_bases == alice_bases,
        alice_bits,
        rng.integers(0, 2, size=key_length),
    )

    bob_bases = alice_bases.copy()  # matched-bases-only assumption
    bob_bits = np.where(
        eve_bases == bob_bases,
        eve_bits,
        rng.integers(0, 2, size=key_length),
    )

    mismatched_bits = int(np.sum(alice_bits != bob_bits))
    qber = mismatched_bits / key_length

    return alice_bits, bob_bits, qber, mismatched_bits


def generate_dataset(
    key_length: int = 8,
    keys_per_batch: int = 100,
    num_batches: int = 100,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Build the full QBER dataset: num_batches x keys_per_batch rows.
    """
    rng = np.random.default_rng(seed)
    records = []

    for batch_id in range(num_batches):
        for key_id in range(keys_per_batch):
            _, _, qber, mismatched_bits = generate_key_with_ir_attack(key_length, rng)
            records.append(
                {
                    "batch_id": batch_id,
                    "key_id": key_id,
                    "key_length": key_length,
                    "mismatched_bits": mismatched_bits,
                    "qber": qber,
                    "eve_present": 1,
                }
            )

    return pd.DataFrame(records)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a BB84 QBER dataset with Eve present (Intercept-Resend attack)."
    )
    parser.add_argument("--key-length", type=int, default=8, help="Bits per key (default: 8)")
    parser.add_argument("--keys-per-batch", type=int, default=100, help="Keys per batch (default: 100)")
    parser.add_argument("--num-batches", type=int, default=100, help="Number of batches (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility (default: 42)")
    parser.add_argument(
        "--out", type=str, default="bb84_qber_dataset.csv", help="Output CSV path"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = generate_dataset(
        key_length=args.key_length,
        keys_per_batch=args.keys_per_batch,
        num_batches=args.num_batches,
        seed=args.seed,
    )
    df.to_csv(args.out, index=False)

    print(f"Generated {len(df)} rows ({args.num_batches} batches x {args.keys_per_batch} keys)")
    print(f"Mean QBER: {df['qber'].mean():.4f} | Std QBER: {df['qber'].std():.4f}")
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()