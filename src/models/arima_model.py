"""ARIMA forecast model wrapping statsmodels."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    """ARIMA(p, d, q) forecast model.

    The model is re-fitted from scratch on each fold's training window.
    No parameters are shared across folds.

    Parameters
    ----------
    order:
        (p, d, q) tuple passed to statsmodels ARIMA.
    """

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1)) -> None:
        self.order = order
        self._result = None

    def fit(self, train_series: pd.Series) -> "ARIMAModel":
        """Fit ARIMA on train_series only."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(train_series.dropna(), order=self.order)
            self._result = model.fit()
        return self

    def predict(self, steps: int) -> np.ndarray:
        """Produce an h-step-ahead forecast.

        Parameters
        ----------
        steps:
            Number of steps to forecast ahead.

        Returns
        -------
        np.ndarray of length `steps`.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before predict().")
        forecast = self._result.forecast(steps=steps)
        return np.asarray(forecast)

    def predict_fold(
        self,
        train: pd.Series,
        horizons: list[int],
    ) -> dict[int, float]:
        """Fit on train, predict at each horizon in one forward pass.

        Returns
        -------
        {horizon: scalar_prediction}
        """
        self.fit(train)
        max_h = max(horizons)
        forecasts = self.predict(max_h)
        return {h: float(forecasts[h - 1]) for h in horizons}
