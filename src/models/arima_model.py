"""ARIMA forecast model wrapping statsmodels.

Used as a univariate benchmark for Paper C's PM10 forecasting protocol.
The model is refit from scratch on each rolling-origin fold's training
window. No parameters are shared across folds and no data beyond the
training window is accessed (strict train-only logic).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    """ARIMA(p, d, q) forecast model.

    Parameters
    ----------
    order:
        (p, d, q) tuple passed to statsmodels ARIMA. Default (2, 1, 2)
        is a conservative, defendable starter configuration for hourly
        PM10 series: a first-difference removes the non-stationary
        component, while the AR(2)/MA(2) structure captures short-term
        autocorrelation without requiring fragile seasonal terms.
    """

    def __init__(self, order: tuple[int, int, int] = (2, 1, 2)) -> None:
        self.order = order
        self._result = None

    def fit(self, train_series: pd.Series) -> "ARIMAModel":
        """Fit ARIMA on train_series only (no test-window contamination)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(train_series.dropna(), order=self.order)
            self._result = model.fit()
        return self

    def predict(self, steps: int) -> np.ndarray:
        """Produce a multi-step-ahead forecast of length `steps`."""
        if self._result is None:
            raise RuntimeError("Call fit() before predict().")
        forecast = self._result.forecast(steps=steps)
        return np.asarray(forecast)

    def predict_fold(
        self,
        train: pd.Series,
        horizons: list[int],
    ) -> dict[int, float]:
        """Fit on train, return {horizon: scalar_prediction}.

        A single forecast of length max(horizons) is produced and the
        requested horizons are sliced from it, so the model is only fit
        once per fold.
        """
        self.fit(train)
        max_h = max(horizons)
        forecasts = self.predict(max_h)
        return {h: float(forecasts[h - 1]) for h in horizons}
