"""
bb84_qber_classifier.py

Trains and compares classifiers that predict the poisoning level
(poison_label: 0/10/20/30/40/50 %) of a BB84 QBER batch from its
distribution features (see bb84_qber_feature_extraction.py).

Models compared:
    - Random Forest
    - Gradient Boosting
    - Logistic Regression (baseline)

For each model, reports:
    - Test-set accuracy
    - Per-class precision/recall/F1 (classification report)
    - Confusion matrix (saved as an image)

Also saves a feature-importance plot from the Random Forest model.

Usage:
    python bb84_qber_classifier.py
    python bb84_qber_classifier.py --input qber_distribution_features.csv --test-size 0.2 --seed 42
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def load_data(input_csv: str):
    df = pd.read_csv(input_csv)
    feature_cols = [c for c in df.columns if c not in ("batch_id", "poison_label")]
    X = df[feature_cols].to_numpy()
    y = df["poison_label"].to_numpy()
    return X, y, feature_cols


def train_and_evaluate(name, model, X_train, X_test, y_train, y_test, class_labels, needs_scaling=False, scaler=None):
    if needs_scaling:
        X_train_use = scaler.transform(X_train)
        X_test_use = scaler.transform(X_test)
    else:
        X_train_use = X_train
        X_test_use = X_test

    model.fit(X_train_use, y_train)
    y_pred = model.predict(X_test_use)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'=' * 60}")
    print(f"{name} — Test Accuracy: {acc:.4f}")
    print(f"{'=' * 60}")
    print(classification_report(y_test, y_pred, labels=class_labels, digits=3))

    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    return model, acc, cm


def plot_confusion_matrices(results, class_labels, out_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (name, _, acc, cm) in zip(axes, results):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nAccuracy = {acc:.3f}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrices saved -> {out_path}")


def plot_feature_importance(model, feature_cols, out_path):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(9, 5))
    plt.bar(range(len(importances)), importances[order], color="steelblue")
    plt.xticks(range(len(importances)), [feature_cols[i] for i in order], rotation=60, ha="right", fontsize=8)
    plt.ylabel("Importance")
    plt.title("Random Forest — Feature Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Feature importance plot saved -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and compare classifiers on the QBER distribution feature table."
    )
    parser.add_argument("--input", type=str, default="qber_distribution_features.csv",
                         help="Path to the feature table CSV")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data held out for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cm-out", type=str, default="confusion_matrices.png",
                         help="Output path for confusion matrix comparison plot")
    parser.add_argument("--fi-out", type=str, default="feature_importance.png",
                         help="Output path for feature importance plot")
    return parser.parse_args()


def main():
    args = parse_args()

    X, y, feature_cols = load_data(args.input)
    class_labels = sorted(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)} | Classes: {class_labels}")

    scaler = StandardScaler().fit(X_train)

    results = []

    rf = RandomForestClassifier(n_estimators=300, random_state=args.seed)
    rf_model, rf_acc, rf_cm = train_and_evaluate(
        "Random Forest", rf, X_train, X_test, y_train, y_test, class_labels
    )
    results.append(("Random Forest", rf_model, rf_acc, rf_cm))

    gb = GradientBoostingClassifier(n_estimators=200, random_state=args.seed)
    gb_model, gb_acc, gb_cm = train_and_evaluate(
        "Gradient Boosting", gb, X_train, X_test, y_train, y_test, class_labels
    )
    results.append(("Gradient Boosting", gb_model, gb_acc, gb_cm))

    lr = LogisticRegression(max_iter=2000, random_state=args.seed)
    lr_model, lr_acc, lr_cm = train_and_evaluate(
        "Logistic Regression", lr, X_train, X_test, y_train, y_test, class_labels,
        needs_scaling=True, scaler=scaler
    )
    results.append(("Logistic Regression", lr_model, lr_acc, lr_cm))

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for name, _, acc, _ in results:
        print(f"{name:<25} Accuracy: {acc:.4f}")

    plot_confusion_matrices(results, class_labels, args.cm_out)
    plot_feature_importance(rf_model, feature_cols, args.fi_out)


if __name__ == "__main__":
    main()