"""
Train a Neural Additive Model on the credit-card marketing dataset.

This script demonstrates the full workflow:
  1. Load and preprocess the CSV (caller's responsibility)
  2. Train the NAMClassifier (library's responsibility)
  3. Evaluate on a held-out test set
  4. Save the trained model + scaler for later use

Place the CSV at examples/data/creditcardmarketing-bbm.csv before running.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

from explainable_nam import NAMClassifier


# Paths - all relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "examples" / "data" / "creditcardmarketing-bbm.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def load_and_preprocess():
    """Load the CSV and return (X_train, X_test, y_train, y_test, feature_names, scaler)."""
    df = pd.read_csv(DATA_PATH)

    df.drop(
        columns=["index", "Q1 Balance", "Q2 Balance", "Q3 Balance",
                 "Q4 Balance", "Customer Number"],
        inplace=True,
    )

    # Separate target
    y = (df["Offer Accepted"] == "Yes").astype(np.float32).values
    df = df.drop(columns=["Offer Accepted"])

    print(f"Target balance: {int(y.sum())} positive / {int((y == 0).sum())} negative "
          f"({100 * y.mean():.1f}% positive)")
    print("Note: this dataset is heavily imbalanced. Use a low threshold or "
          "class weighting for production use.\n")

    # Fill missing balances
    df["Average Balance"] = df["Average Balance"].fillna(df["Average Balance"].median())

    # Encode categoricals
    for col in ["Overdraft Protection", "Own Your Home"]:
        df[col] = (df[col] == "Yes").astype(int)

    ordinal_map = {"Low": 0, "Medium": 1, "High": 2}
    df["Income Level"] = df["Income Level"].map(ordinal_map)
    df["Credit Rating"] = df["Credit Rating"].map(ordinal_map)

    df = pd.get_dummies(df, columns=["Reward", "Mailer Type"], drop_first=True)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    feature_names = df.columns.tolist()
    X = df.values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train, y_test, feature_names, scaler


def main():
    print(f"Loading data from {DATA_PATH}")
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess()
    print(f"Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")
    print(f"Features ({len(feature_names)}): {feature_names}\n")

    # Train the NAM
    model = NAMClassifier(
        hidden_size=32,
        n_layers=3,
        epochs=50,
        batch_size=256,
        verbose=True,
    )
    model.fit(X_train, y_train)

    # Evaluate
    probs = model.predict_proba(X_test)
    preds = (probs >= 0.5).astype(int)
    print(f"\nTest AUC:      {roc_auc_score(y_test, probs):.3f}")
    print(f"Test accuracy: {accuracy_score(y_test, preds):.1%}")
    print(f"Positives predicted: {preds.sum()} of {len(preds)} "
          f"(at default threshold 0.5)\n")

    # Save the model and scaler for the scoring example
    model_path = OUTPUTS_DIR / "nam_credit_model.pt"
    scaler_path = OUTPUTS_DIR / "nam_credit_scaler.pkl"
    torch.save(model.model_.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "feature_names": feature_names}, f)

    print(f"Saved model to  {model_path}")
    print(f"Saved scaler to {scaler_path}")


if __name__ == "__main__":
    main()
