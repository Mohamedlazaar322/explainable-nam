# explainable-nam

A Python library implementing Neural Additive Models (NAM) for binary classification on tabular data, with mathematically exact per-feature explanations.


---

## Why this exists

Standard neural networks require post-hoc methods (SHAP, LIME, Integrated Gradients) to produce explanations, and those explanations are mathematical approximations of the model's reasoning; not exact decompositions. For regulated domains (credit scoring, insurance pricing) where decisions must be defensible to a regulator under GDPR Art. 22 and the EU AI Act, this approximation introduces a real risk.

A Neural Additive Model has a different structure:

```
prediction = f₁(x₁) + f₂(x₂) + ... + fₙ(xₙ) + bias
```

Each `fᵢ` is a small neural network learning a non-linear shape for one feature. Because features never mix, the per-feature contribution to any decision is exact and reproducible.

## Installation

```bash
git clone <your-repo-url>
cd explainable-nam
pip install -e .
```

Requires Python 3.9+, PyTorch 1.12+, NumPy, Matplotlib.

## Quick start

```python
from explainable_nam import NAMClassifier
import numpy as np

# X must be pre-processed: numeric, no NaN, standardized
# y must be 0/1 binary labels
model = NAMClassifier(hidden_size=32, n_layers=3, epochs=50)
model.fit(X_train, y_train)

# Predictions
probs = model.predict_proba(X_test)
preds = model.predict(X_test, threshold=0.5)

# Exact per-feature contributions
contributions = model.explain(X_test)
# Property: sum(contributions[i]) + model.bias() == model logit, exactly.

# Global view: shape functions for the most influential features
model.plot_shape_functions(feature_names, top_n=6, save_path="shapes.png")
```

## What the library handles

- The NAM architecture (per-feature subnetworks summed at the end)
- Training with Adam, BCE loss, mini-batches
- Prediction and exact per-feature contribution extraction
- Shape function plotting over the actual training-data range
- Defensive checks (NaN/inf inputs, feature-count mismatch, fit-before-predict)

## What the library does NOT handle

These are intentionally the caller's responsibility, matching the design philosophy of PyTorch itself:

- Data loading and cleaning
- Missing value imputation
- Categorical encoding (one-hot, ordinal, target encoding)
- Feature scaling (StandardScaler, MinMaxScaler, etc.)
- Class imbalance handling (thresholding, resampling, class weights)
- Model save/load — use `torch.save(model.model_.state_dict(), path)`
- Train/test splitting
- Compliance documentation generation
- Audit logging

## How this differs from existing explanation tools

Standard practice today combines a black-box model with a post-hoc explainer:

- **SHAP** — approximates contributions via cooperative game theory
- **LIME** — approximates the model locally with a simpler model
- **Integrated Gradients (Captum)** — approximates contributions by integrating gradients

All three are *post-hoc approximations*: the sum of attributed contributions does not exactly reconstruct the model's output, and the choice of baseline or sampling can change the explanation.

A NAM is *interpretable by construction*. Nothing is approximated, the same input always produces the same explanation, and no additional library is needed at inference time.

The empirical difference is one number — see `examples/baseline_comparison.py`.

## Examples

The `examples/` folder contains three runnable scripts:

| File | What it does |
|---|---|
| `train_credit_marketing.py` | Trains a NAM on a credit-card marketing dataset, saves model + scaler |
| `score_one_customer.py` | Loads the saved model and produces a GDPR Art. 22 / EU AI Act-compliant decision notice for one customer |

Run from the project root:

```bash
python examples/train_credit_marketing.py
python examples/score_one_customer.py
```

A sample output is saved at `outputs/sample_decision_output.txt`.

## Sample output

For a single customer scored by `score_one_customer.py`:

```
===========================================================================
AUTOMATED DECISION NOTICE
===========================================================================
Decision date:      2026-06-02 14:23:11
Decision:           DECLINE
Confidence (prob):  8.4%
Decision threshold: 15%

PRINCIPAL REASONS  (ECOA / Regulation B - top 4, ranked)
---------------------------------------------------------------------------
Principal reasons for decline:
  1. Mailer Type_Postcard      Observed: 0    Contribution: -0.482
  2. Credit Rating             Observed: 0    Contribution: -0.301
  3. Income Level              Observed: 0    Contribution: -0.205
  4. Average Balance           Observed: 412  Contribution: -0.176

Math verification:
  sum(contributions) + bias = -2.3915
  model logit reconstructed = -2.3915   (match confirms exactness)
===========================================================================
```

The last two lines are the key property: the explanation IS the model's computation, not a story constructed after the fact.

## API reference

The library exposes one class: `NAMClassifier`.

| Method | Returns | Description |
|---|---|---|
| `fit(X, y)` | self | Trains the NAM. `X` must be numeric and 2D; `y` must be 0/1 |
| `predict_proba(X)` | array of shape `(n,)` | Probability of the positive class |
| `predict(X, threshold)` | array of shape `(n,)` | Binary predictions (0/1) |
| `explain(X)` | array of shape `(n, n_features)` | Exact per-feature contributions |
| `bias()` | float | The model's bias term (constant added to every prediction) |
| `shape_function(idx, x_min, x_max)` | `(x, y)` arrays | Learned shape for one feature |
| `plot_shape_functions(names, top_n)` | matplotlib Figure | Plots the top features' learned shapes |

Constructor parameters: `hidden_size`, `n_layers`, `epochs`, `learning_rate`, `batch_size`, `weight_decay`, `device`, `random_state`, `verbose`. All have sensible defaults; tune as needed.

## Limitations

- Binary classification only (no multi-class, no regression)
- No feature-interaction modeling (additive by design)
- Python loop over features in the forward pass; slow for 200+ features

## References

- Agarwal et al., *"Neural Additive Models: Interpretable Machine Learning with Neural Nets"* (NeurIPS 2021)


## License

MIT
