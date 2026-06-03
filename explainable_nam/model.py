"""
explainable_nam - a small library for Neural Additive Models.

A NAM has one small neural network per feature; the final score is the sum
of each feature's contribution. This makes the per-feature contribution
EXACT rather than approximated, so the model is interpretable by construction.

The library handles only the model: training, prediction, explanation,
and shape-function plotting. Data cleaning, missing values, categorical
encoding, save/load, logging, and business reporting are the caller's
responsibility.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Internal building blocks

class _FeatureNet(nn.Module):
    """One small subnetwork that handles a single feature."""

    def __init__(self, hidden: int = 32, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _NAM(nn.Module):
    """Sum of per-feature subnetworks + bias."""

    def __init__(self, n_features: int, hidden: int = 32, n_layers: int = 3):
        super().__init__()
        self.feature_nets = nn.ModuleList(
            [_FeatureNet(hidden, n_layers) for _ in range(n_features)]
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, return_contributions: bool = False):
        contributions = torch.stack(
            [self.feature_nets[i](x[:, i:i + 1]) for i in range(x.shape[1])],
            dim=1,
        ).squeeze(-1)
        output = contributions.sum(dim=1, keepdim=True) + self.bias
        if return_contributions:
            return output, contributions
        return output


# Public class -- the NAMClassifier that users will interact with

class NAMClassifier:
    """

    hidden_size : int
        Hidden units per layer inside each feature subnetwork.
    n_layers : int
        Number of hidden layers in each feature subnetwork.
    epochs : int
        Number of training epochs.
    learning_rate : float
        Adam learning rate.
    batch_size : int
        Mini-batch size during training.
    weight_decay : float
        L2 regularization strength.
    device : str or None
        "cuda", "cpu", or None to auto-detect.
    random_state : int or None
        Seed for reproducibility.

    Usage
    -----
        model = NAMClassifier()
        model.fit(X_train, y_train)
        prob = model.predict_proba(x)
        contribs = model.explain(x)
        model.plot_shape_functions(feature_names)
    """

    def __init__(
        self,
        hidden_size: int = 32,
        n_layers: int = 3,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
        random_state: Optional[int] = 42,
        verbose: bool = True,
    ):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.verbose = verbose

        self.model_: Optional[_NAM] = None
        self.n_features_: Optional[int] = None
        # Store the actual training-data range per feature so shape functions
        # can be plotted over the real data range, not a guessed [-3, 3].
        self.feature_min_: Optional[np.ndarray] = None
        self.feature_max_: Optional[np.ndarray] = None

    # Training

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NAMClassifier":
        """Train the NAM on numeric, already-scaled features.

        X : shape (n_samples, n_features), float
        y : shape (n_samples,) in {0, 1}
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Defensive checks - fail loudly instead of silently producing garbage.
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D (n_samples, n_features), got shape {X.shape}."
            )
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError(
                "X contains NaN or infinite values. Handle missing values "
                "before calling fit()."
            )

        self.n_features_ = X.shape[1]
        # Remember the actual training-data range for honest plotting later.
        self.feature_min_ = X.min(axis=0)
        self.feature_max_ = X.max(axis=0)

        self.model_ = _NAM(
            n_features=self.n_features_,
            hidden=self.hidden_size,
            n_layers=self.n_layers,
        ).to(self.device)

        opt = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(y)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model_.train()
        for epoch in range(self.epochs):
            total = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device).unsqueeze(1)
                opt.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item()
            if self.verbose and (epoch + 1) % max(1, self.epochs // 4) == 0:
                print(f"  Epoch {epoch + 1:3d}/{self.epochs}  "
                      f"avg loss = {total / len(loader):.4f}")

        self.model_.eval()
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class for each row of X."""
        X = self._check_and_prepare(X)
        with torch.no_grad():
            logits = self.model_(torch.tensor(X).to(self.device))
            return torch.sigmoid(logits).cpu().numpy().flatten()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction (0 or 1) for each row of X."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------
    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Return the per-feature contributions for each row of X.

        Output shape: (n_samples, n_features). Row sums + model bias equal
        the model's raw output (logit), exactly. Positive contributions push
        toward the positive class; negative push toward the negative class.
        """
        X = self._check_and_prepare(X)
        with torch.no_grad():
            _, contributions = self.model_(
                torch.tensor(X).to(self.device),
                return_contributions=True,
            )
        return contributions.cpu().numpy()

    def bias(self) -> float:
        """The model's bias term, added once to the sum of contributions."""
        self._check_fitted()
        return float(self.model_.bias.item())

    # ------------------------------------------------------------------
    # Shape functions (global view of what each feature learned)
    # ------------------------------------------------------------------
    def shape_function(
        self, feature_index: int,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        n_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_values, contributions) for one feature over a range.

        If x_min/x_max are not provided, the actual range observed in the
        training data is used. This avoids extrapolating into regions the
        model has never seen.
        """
        self._check_fitted()
        if x_min is None:
            x_min = float(self.feature_min_[feature_index])
        if x_max is None:
            x_max = float(self.feature_max_[feature_index])
        x_range = torch.linspace(
            x_min, x_max, n_points).unsqueeze(1).to(self.device)
        with torch.no_grad():
            y = self.model_.feature_nets[feature_index](
                x_range).cpu().numpy().flatten()
        return x_range.cpu().numpy().flatten(), y

    def plot_shape_functions(
        self,
        feature_names: list[str],
        top_n: int = 6,
        save_path: Optional[str] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
    ):
        """Plot the learned shape function for the top N features by range.

        If x_min/x_max are not provided, each feature is plotted over its
        actual range observed in the training data.
        """
        self._check_fitted()
        if len(feature_names) != self.n_features_:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries but model "
                f"was fit with {self.n_features_} features."
            )
        import matplotlib.pyplot as plt

        # Pick features whose contribution varies the most across their range.
        ranges = []
        for i in range(self.n_features_):
            _, y = self.shape_function(i, x_min, x_max)
            ranges.append((i, y.max() - y.min()))
        ranges.sort(key=lambda t: -t[1])
        top_indices = [i for i, _ in ranges[:top_n]]

        cols = 3
        rows = (top_n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
        axes = np.atleast_1d(axes).flatten()
        for ax, idx in zip(axes, top_indices):
            x, y = self.shape_function(idx, x_min, x_max)
            ax.plot(x, y, linewidth=2)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.set_title(feature_names[idx], fontsize=11)
            ax.set_xlabel("feature value", fontsize=9)
            ax.set_ylabel("contribution", fontsize=9)
            ax.grid(alpha=0.3)
        for ax in axes[len(top_indices):]:
            ax.axis("off")

        plt.suptitle("Learned shape functions (top features)", fontsize=13)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    def _check_fitted(self):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call .fit(X, y) first.")

    def _check_and_prepare(self, X: np.ndarray) -> np.ndarray:
        """Validate input X and convert it to the right shape and dtype."""
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fit with "
                f"{self.n_features_} features."
            )
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError(
                "X contains NaN or infinite values. Handle missing values "
                "before calling this method."
            )
        return X
