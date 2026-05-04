"""LightGBM forecast model with lag-based features.

Uses a direct (per-horizon) forecasting strategy: a separate LightGBM
regressor is trained for each forecast horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb


class LGBMModel:
    """Gradient-boosted trees forecaster with lag features.

    Parameters
    ----------
    n_estimators:
        Number of boosting rounds per horizon model.
    max_depth:
        Maximum tree depth.
    n_lags:
        Number of autoregressive lag features (lags 1 … n_lags).
    horizons:
        Forecast horizons; a separate model is trained for each.
    num_leaves:
        Maximum number of leaves per tree. ``None`` falls back to the
        LightGBM default (31).
    learning_rate:
        Boosting shrinkage. ``None`` keeps the LightGBM default (0.1).
    n_jobs:
        Threads per booster. Defaults to 1 to keep runs deterministic on
        small machines.
    random_state:
        Seed propagated to LightGBM for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        n_lags: int = 24,
        horizons: list[int] | None = None,
        num_leaves: int | None = None,
        learning_rate: float | None = None,
        n_jobs: int = 1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_lags = n_lags
        self.horizons = sorted(horizons or [1, 6, 12, 24])
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._models: dict[int, lgb.Booster] = {}

    def _make_features(
        self, values: np.ndarray, horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build lag-feature matrix X and target vector y for given horizon."""
        n = len(values)
        rows_X, rows_y = [], []
        for i in range(self.n_lags, n - horizon + 1):
            features = values[i - self.n_lags : i][::-1]  # lag-1 first
            target = values[i + horizon - 1]
            rows_X.append(features)
            rows_y.append(target)
        if not rows_X:
            raise ValueError("Training series too short for the given n_lags and horizon.")
        return np.array(rows_X), np.array(rows_y)

    def fit(self, train_series: pd.Series) -> "LGBMModel":
        """Train one LightGBM model per horizon.

        The model is fitted exclusively on train_series (no leakage).
        Random state is fixed to 42 for reproducibility.
        """
        values = train_series.dropna().values.astype(float)
        self._models = {}

        for h in self.horizons:
            X, y = self._make_features(values, h)
            params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "verbosity": -1,
            }
            if self.num_leaves is not None:
                params["num_leaves"] = self.num_leaves
            if self.learning_rate is not None:
                params["learning_rate"] = self.learning_rate
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            self._models[h] = model

        return self

    def predict(self, context: np.ndarray, horizon: int) -> float:
        """Predict h steps ahead from a lag context.

        Parameters
        ----------
        context:
            At least `n_lags` most-recent observations (scaled).
        horizon:
            Must be in self.horizons.

        Returns
        -------
        float
        """
        if horizon not in self._models:
            raise ValueError(
                f"horizon {horizon} not in trained horizons {self.horizons}. "
                "Call fit() first."
            )
        ctx = np.asarray(context, dtype=float)[-self.n_lags:][::-1]
        return float(self._models[horizon].predict(ctx.reshape(1, -1))[0])
