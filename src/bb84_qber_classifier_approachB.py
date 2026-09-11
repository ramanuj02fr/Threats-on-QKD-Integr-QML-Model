"""
Classifier training/comparison script for Approach B (stealthy bounded perturbation).

Loads the batch-wise distribution features (600 rows, 13 features),
does an 80-20 stratified train-test split on poison_label, and
compares 3 models: Random Forest, Gradient Boosting, Logistic Regression.

Prints for each model: accuracy, classification report (precision/
recall/f1 per class), and confusion matrix. Also prints feature
importance for the tree-based models (Random Forest, Gradient Boosting).

Input : poisoned_datasets/qber_distribution_features_approachB.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- Config ----------------
INPUT_CSV = Path("poisoned_datasets/qber_distribution_features_approachB.csv")
RANDOM_SEED = 42
TEST_SIZE = 0.2


def load_data():
    df = pd.read_csv(INPUT_CSV)
    feature_cols = [c for c in df.columns if c not in ("batch_id", "poison_label")]
    X = df[feature_cols]
    y = df["poison_label"]
    return X, y, feature_cols


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'=' * 60}")
    print(f"Model: {name}")
    print(f"{'=' * 60}")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    return acc, model


def print_feature_importance(name, model, feature_cols):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances
        }).sort_values("importance", ascending=False)
        print(f"\nFeature importance ({name}):")
        print(imp_df.to_string(index=False))


def main():
    X, y, feature_cols = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    print(f"Train class distribution:\n{y_train.value_counts().sort_index()}")

    # Logistic Regression needs scaled features; tree models don't.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    acc_rf, rf_fitted = evaluate_model("Random Forest", rf, X_train, X_test, y_train, y_test)
    results["Random Forest"] = acc_rf
    print_feature_importance("Random Forest", rf_fitted, feature_cols)

    gb = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
    acc_gb, gb_fitted = evaluate_model("Gradient Boosting", gb, X_train, X_test, y_train, y_test)
    results["Gradient Boosting"] = acc_gb
    print_feature_importance("Gradient Boosting", gb_fitted, feature_cols)

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    acc_lr, _ = evaluate_model("Logistic Regression", lr, X_train_scaled, X_test_scaled, y_train, y_test)
    results["Logistic Regression"] = acc_lr

    print(f"\n{'=' * 60}")
    print("SUMMARY (Approach B - stealthy bounded perturbation)")
    print(f"{'=' * 60}")
    for name, acc in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"{name:25s}: {acc:.4f}")


if __name__ == "__main__":
    main()