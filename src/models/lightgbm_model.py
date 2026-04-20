"""LightGBM direct multi-horizon forecast model for Paper C.

One LightGBM regressor is trained per horizon and per fold, strictly on
the training window of each rolling-origin fold (no data leakage).

Feature engineering
-------------------
A minimal autoregressive feature vector is built at each position t from
lagged PM10 values only: lags [1, 2, 3, 6, 12, 24], i.e.

    X[t] = [pm10[t],   pm10[t-1], pm10[t-2],
            pm10[t-5], pm10[t-11], pm10[t-23]]

Direct forecasting strategy
---------------------------
For horizon h, training pairs (X[t], pm10[t+h]) are constructed from all
valid positions inside the training window. Prediction at the fold origin
uses the feature vector built from the last ``max_lag`` observations of
the training series, which are all available without touching the test
window.

LightGBM configuration
-----------------------
Conservative, memory-friendly hyperparameters designed for Mac mini M2
with 8 GB RAM. No grid search. No auto-ML.

    n_estimators : 300   — enough capacity without long training time
    num_leaves   : 31    — default; controls model complexity
    max_depth    : 5     — caps tree depth for regularisation
    learning_rate: 0.05  — modest rate; paired with 300 trees
    subsample    : 0.8   — row subsampling (reduces variance)
    colsample_bytree: 0.8 — feature subsampling
    min_child_samples: 20 — avoids overfitting on small leaf nodes
    n_jobs       : 1     — single thread (prevents memory spikes)
    random_state : 42    — reproducibility
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb


LAGS = [1, 2, 3, 6, 12, 24]
MAX_LAG = max(LAGS)

LGBM_PARAMS: dict = {
    "n_estimators": 300,
    "num_leaves": 31,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "n_jobs": 1,
    "random_state": 42,
    "verbose": -1,
}


def _build_features(values: np.ndarray, pos: int) -> np.ndarray:
    """Build the lag feature vector at position ``pos``.

    Returns a 1-D array of length len(LAGS). Requires pos >= MAX_LAG - 1.
    """
    return np.array([values[pos - (lag - 1)] for lag in LAGS], dtype=float)


def _build_training_dataset(
    values: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Construct (X, y) for one horizon from the training window.

    Valid sample positions: t in [MAX_LAG - 1, len(values) - horizon - 1].
    Target: values[t + horizon].
    """
    n = len(values)
    rows_X, rows_y = [], []
    for t in range(MAX_LAG - 1, n - horizon):
        rows_X.append(_build_features(values, t))
        rows_y.append(values[t + horizon])
    if not rows_X:
        return np.empty((0, len(LAGS))), np.empty(0)
    return np.array(rows_X, dtype=float), np.array(rows_y, dtype=float)


class LightGBMModel:
    """One LightGBM regressor per forecast horizon.

    Parameters
    ----------
    params:
        LightGBM regressor parameters. Defaults to ``LGBM_PARAMS``.
    horizons:
        Forecast horizons in steps. Must be provided at construction or
        at ``fit`` time.
    """

    def __init__(
        self,
        params: dict | None = None,
        horizons: list[int] | None = None,
    ) -> None:
        self.params = params if params is not None else LGBM_PARAMS.copy()
        self.horizons = horizons or [1, 6, 12, 24]
        self._models: dict[int, lgb.LGBMRegressor] = {}

    def fit(self, train_values: np.ndarray) -> "LightGBMModel":
        """Fit one regressor per horizon on ``train_values``.

        Parameters
        ----------
        train_values:
            1-D NumPy array of training observations (train window only).
        """
        self._models = {}
        for h in self.horizons:
            X, y = _build_training_dataset(train_values, h)
            if len(X) < 2:
                warnings.warn(
                    f"LightGBM h={h}: training dataset too small "
                    f"({len(X)} samples). Model will not be fit for this horizon."
                )
                continue
            model = lgb.LGBMRegressor(**self.params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
            self._models[h] = model
        return self

    def predict_at_origin(
        self, train_values: np.ndarray, horizons: list[int]
    ) -> dict[int, float | None]:
        """Return point forecasts for each horizon from the fold origin.

        The feature vector is built from the last ``MAX_LAG`` positions of
        ``train_values`` — all within the training window.

        Returns
        -------
        {horizon: predicted_value_or_None}
            None means the model for that horizon was not fit (training set
            too small), and the caller should apply a fallback.
        """
        if len(train_values) < MAX_LAG:
            return {h: None for h in horizons}

        origin_pos = len(train_values) - 1
        x_origin = _build_features(train_values, origin_pos).reshape(1, -1)

        results: dict[int, float | None] = {}
        for h in horizons:
            if h not in self._models:
                results[h] = None
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = self._models[h].predict(x_origin)
                results[h] = float(pred[0])
        return results

    def fit_predict_fold(
        self,
        train_values: np.ndarray,
        horizons: list[int],
    ) -> dict[int, float | None]:
        """Convenience: fit then predict in one call."""
        self.fit(train_values)
        return self.predict_at_origin(train_values, horizons)
