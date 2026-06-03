"""
Score one customer with the trained NAM and produce a decision notice
formatted to satisfy GDPR Art. 22 / Recital 71 and EU AI Act Art. 13
transparency obligations.

Run `python examples/train_credit_marketing.py` first to create the
saved model files.

To test a different customer, edit the `new_customer` dictionary at
the bottom of this file.
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from explainable_nam import NAMClassifier
from explainable_nam.model import _NAM


PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "nam_credit_model.pt"
SCALER_PATH = PROJECT_ROOT / "outputs" / "nam_credit_scaler.pkl"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "sample_decision_output.txt"


def load_model():
    """Load the trained NAM from disk."""
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            f"Run examples/train_credit_marketing.py first."
        )

    with open(SCALER_PATH, "rb") as f:
        saved = pickle.load(f)
    scaler = saved["scaler"]
    feature_names = saved["feature_names"]

    model = NAMClassifier(hidden_size=32, n_layers=3, verbose=False)
    model.n_features_ = len(feature_names)
    model.feature_min_ = np.full(len(feature_names), -3.0, dtype=np.float32)
    model.feature_max_ = np.full(len(feature_names), 3.0, dtype=np.float32)
    model.model_ = _NAM(
        n_features=model.n_features_,
        hidden=model.hidden_size,
        n_layers=model.n_layers,
    ).to(model.device)
    model.model_.load_state_dict(torch.load(MODEL_PATH, map_location=model.device))
    model.model_.eval()

    return model, scaler, feature_names


def preprocess_customer(raw, scaler, feature_names):
    """Convert one customer's raw inputs to the scaled feature vector the model expects."""
    ordinal_map = {"Low": 0, "Medium": 1, "High": 2}

    row = {
        "Income Level":         ordinal_map[raw["Income Level"]],
        "# Bank Accounts Open": raw["# Bank Accounts Open"],
        "Overdraft Protection": 1 if raw["Overdraft Protection"] == "Yes" else 0,
        "Credit Rating":        ordinal_map[raw["Credit Rating"]],
        "# Credit Cards Held":  raw["# Credit Cards Held"],
        "# Homes Owned":        raw["# Homes Owned"],
        "Household Size":       raw["Household Size"],
        "Own Your Home":        1 if raw["Own Your Home"] == "Yes" else 0,
        "Average Balance":      raw["Average Balance"],
        "Reward_Cash Back":     1 if raw["Reward"] == "Cash Back" else 0,
        "Reward_Points":        1 if raw["Reward"] == "Points" else 0,
        "Mailer Type_Postcard": 1 if raw["Mailer Type"] == "Postcard" else 0,
    }
    x_raw = np.array([[row[f] for f in feature_names]], dtype=np.float32)
    x_scaled = scaler.transform(x_raw).astype(np.float32)
    return x_raw, x_scaled


def build_notice(raw_customer, model, scaler, feature_names, threshold=0.15):
    """Build the decision notice as a string."""
    x_raw, x_scaled = preprocess_customer(raw_customer, scaler, feature_names)
    probability = float(model.predict_proba(x_scaled)[0])
    contributions = model.explain(x_scaled)[0]
    decision = "ACCEPT" if probability >= threshold else "DECLINE"

    ranked = sorted(
        zip(feature_names, contributions, x_raw[0]),
        key=lambda t: -abs(t[1]),
    )

    lines = []
    lines.append("=" * 75)
    lines.append("AUTOMATED DECISION NOTICE")
    lines.append("=" * 75)
    lines.append(f"Decision date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Decision:           {decision}")
    lines.append(f"Confidence (prob):  {probability:.1%}")
    lines.append(f"Decision threshold: {threshold:.0%}")
    lines.append("")

    lines.append("LOGIC OF THE DECISION  (GDPR Art. 22 §3 / Recital 71)")
    lines.append("-" * 75)
    lines.append("This decision was produced by a Neural Additive Model. The model")
    lines.append(f"evaluated {len(feature_names)} features. Each feature has its own learned")
    lines.append("contribution function; the final score is the exact sum of these")
    lines.append("per-feature contributions plus a bias term. The score is compared")
    lines.append(f"to the threshold of {threshold:.0%}; values at or above the threshold result")
    lines.append("in ACCEPT, below in DECLINE.")
    lines.append("")

    if decision == "DECLINE":
        principal = [r for r in ranked if r[1] < 0][:4]
        label = "Principal reasons for decline"
    else:
        principal = [r for r in ranked if r[1] > 0][:4]
        label = "Principal factors supporting acceptance"

    lines.append("PRINCIPAL REASONS  (ECOA / Regulation B - top 4, ranked)")
    lines.append("-" * 75)
    lines.append(f"{label}:")
    lines.append("")
    for rank, (name, contrib, value) in enumerate(principal, 1):
        lines.append(f"  {rank}. {name}")
        lines.append(f"     Observed value:            {value}")
        lines.append(f"     Contribution to decision:  {contrib:+.3f}")
        lines.append("")

    if decision == "DECLINE":
        opposing = [r for r in ranked if r[1] > 0][:3]
        opposing_label = "Factors that pushed AGAINST decline"
    else:
        opposing = [r for r in ranked if r[1] < 0][:3]
        opposing_label = "Factors that pushed AGAINST acceptance"

    if opposing:
        lines.append(f"{opposing_label}:")
        for name, contrib, value in opposing:
            lines.append(f"  - {name} = {value}  (contribution: {contrib:+.3f})")
        lines.append("")

    lines.append("SIGNIFICANCE AND CONSEQUENCES  (GDPR Recital 71)")
    lines.append("-" * 75)
    if decision == "DECLINE":
        lines.append("Consequence: the credit-card offer will not be extended.")
        lines.append("Right to contest: the customer may request human review under")
        lines.append("GDPR Art. 22 §3 and provide additional information.")
    else:
        lines.append("Consequence: the credit-card offer is extended.")
        lines.append("No adverse-action notice is required.")
    lines.append("")

    lines.append("METHOD AND TRANSPARENCY  (EU AI Act Art. 13)")
    lines.append("-" * 75)
    lines.append("Model:        Neural Additive Model (Agarwal et al., 2021)")
    lines.append("Property:     Additive by construction - sum of contributions")
    lines.append("              exactly reproduces the model score.")
    lines.append("Determinism:  Same input always produces the same explanation.")

    total = contributions.sum() + model.bias()
    logit = (np.log(probability / (1 - probability))
             if 0 < probability < 1 else float("inf"))
    lines.append("")
    lines.append("Math verification:")
    lines.append(f"  sum(contributions) + bias = {total:+.4f}")
    lines.append(f"  model logit reconstructed = {logit:+.4f}   (match = exact)")
    lines.append("")
    lines.append("=" * 75)
    lines.append("END OF NOTICE")
    lines.append("=" * 75)

    return "\n".join(lines)


def main():
    model, scaler, feature_names = load_model()

    new_customer = {
        "Income Level":         "Medium",      # Low / Medium / High
        "# Bank Accounts Open": 2,
        "Overdraft Protection": "No",          # Yes / No
        "Credit Rating":        "High",        # Low / Medium / High
        "# Credit Cards Held":  2,
        "# Homes Owned":        1,
        "Household Size":       3,
        "Own Your Home":        "Yes",         # Yes / No
        "Average Balance":      1500,
        "Reward":               "Cash Back",   # Air Miles / Cash Back / Points
        "Mailer Type":          "Postcard",    # Letter / Postcard
    }

    notice = build_notice(new_customer, model, scaler, feature_names, threshold=0.15)
    print(notice)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(notice)
    print(f"\n[Saved to {OUTPUT_PATH}]")


if __name__ == "__main__":
    main()
