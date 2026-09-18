"""Phase predictor: multinomial logistic + Cox/AFT survival model.

Predicts:
    - Phase class: DIRECT / GROKKING / MEMORIZED / UNDERFIT / PARTIAL
    - log(T_gen): survival regression

Primary model: Regularized Cox/AFT survival model.
Secondary model: Multinomial logistic for phase classification.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings


class PhasePredictor:
    """Combined phase + survival predictor.

    Uses regularized logistic regression for phase classification
    and a simple AFT (accelerated failure time) model for log(T_gen).

    No sklearn dependency required — uses numpy implementations.
    """

    def __init__(
        self,
        feature_names: List[str],
        lambda_reg: float = 0.01,
        max_iter: int = 1000,
        lr: float = 0.01,
    ):
        self.feature_names = feature_names
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.lr = lr
        self.phase_model = None      # multinomial logistic weights
        self.phase_classes = None
        self.survival_model = None   # AFT weights
        self.is_fitted = False
        self.feature_means = None
        self.feature_stds = None

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Standardize features."""
        if fit:
            self.feature_means = X.mean(axis=0)
            self.feature_stds = X.std(axis=0)
            self.feature_stds[self.feature_stds == 0] = 1.0
        return (X - self.feature_means) / self.feature_stds

    def fit(
        self,
        X: np.ndarray,
        phases: np.ndarray,
        log_T_gen: np.ndarray,
        censored: np.ndarray,
    ) -> "PhasePredictor":
        """Fit the phase + survival predictor.

        Args:
            X: Feature matrix (n_samples, n_features).
            phases: Phase labels (n_samples,).
            log_T_gen: log(T_gen) for uncensored, log(budget) for censored.
            censored: 1 if censored (T_gen not reached), 0 if observed.

        Returns:
            self
        """
        X_std = self._preprocess(X, fit=True)
        n, d = X_std.shape

        # --- Phase classification (multinomial logistic) ---
        self.phase_classes = sorted(np.unique(phases))
        K = len(self.phase_classes)
        class_to_idx = {c: i for i, c in enumerate(self.phase_classes)}

        Y_phase = np.zeros((n, K))
        for i in range(n):
            Y_phase[i, class_to_idx[phases[i]]] = 1.0

        # One-vs-rest binary logistic for each class (simple, robust)
        self.phase_model = np.zeros((K, d + 1))  # +1 for bias
        for k in range(K):
            y_k = Y_phase[:, k]
            w = np.zeros(d + 1)
            X_bias = np.hstack([X_std, np.ones((n, 1))])
            for _ in range(self.max_iter):
                logits = X_bias @ w
                probs = 1 / (1 + np.exp(-logits))
                grad = X_bias.T @ (probs - y_k) / n + self.lambda_reg * np.concatenate([w[:-1], [0]])
                w -= self.lr * grad
            self.phase_model[k] = w

        # --- Survival: AFT model ---
        # log(T_gen) = X @ beta + epsilon
        # For censored observations, use Tobit-style: only use uncensored for OLS
        uncensored = ~censored.astype(bool)
        if uncensored.sum() > 1:
            X_bias = np.hstack([X_std, np.ones((n, 1))])
            X_unc = X_bias[uncensored]
            y_unc = log_T_gen[uncensored]
            # Ridge regression
            A = X_unc.T @ X_unc + self.lambda_reg * np.eye(d + 1)
            A[-1, -1] = X_unc.T @ X_unc[-1, -1]  # don't penalize bias
            b = X_unc.T @ y_unc
            try:
                self.survival_model = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                self.survival_model = np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            self.survival_model = np.zeros(d + 1)

        self.is_fitted = True
        return self

    def predict_phase_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict phase probabilities.

        Returns: (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("PhasePredictor not fitted")
        X_std = self._preprocess(X)
        X_bias = np.hstack([X_std, np.ones((X_std.shape[0], 1))])

        # One-vs-rest -> normalize
        logits = X_bias @ self.phase_model.T  # (n, K)
        # Softmax over OvR logits
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs

    def predict_phase(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely phase."""
        probs = self.predict_phase_proba(X)
        idx = probs.argmax(axis=1)
        return np.array([self.phase_classes[i] for i in idx])

    def predict_log_T_gen(self, X: np.ndarray) -> np.ndarray:
        """Predict log(T_gen) via AFT model."""
        if not self.is_fitted:
            raise RuntimeError("PhasePredictor not fitted")
        X_std = self._preprocess(X)
        X_bias = np.hstack([X_std, np.ones((X_std.shape[0], 1))])
        return X_bias @ self.survival_model

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Full prediction: phase probs + log(T_gen)."""
        return {
            "phase_proba": self.predict_phase_proba(X),
            "phase": self.predict_phase(X),
            "log_T_gen": self.predict_log_T_gen(X),
        }
