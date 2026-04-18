"""Persistence (naive) forecast model."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PersistenceModel:
    """Naive persistence: predict the last observed value for all horizons.

    This model requires no training and serves as the mandatory baseline
    for computing skill scores.
    """

    def fit(self, series: pd.Series) -> "PersistenceModel":
        """No-op: persistence has no learnable parameters."""
        return self

    def predict(self, origin_value: float, horizon: int) -> float:
        """Return the origin value regardless of horizon."""
        return float(origin_value)

    def predict_fold(
        self,
        train: pd.Series,
        test_indices: dict[int, np.ndarray],
        horizons: list[int],
        series_values: np.ndarray,
        origin_pos: int,
    ) -> dict[int, np.ndarray]:
        """Generate persistence predictions for all horizons of one fold.

        Parameters
        ----------
        train:
            Training series for this fold (unused but kept for API symmetry).
        test_indices:
            {horizon: array_of_integer_positions_in_full_series}.
        horizons:
            List of forecast horizons.
        series_values:
            Full series as a numpy array (for reading origin value).
        origin_pos:
            Integer position of the last training observation.

        Returns
        -------
        {horizon: np.ndarray of predictions}
        """
        origin_val = series_values[origin_pos]
        return {h: np.full(len(test_indices.get(h, [])), origin_val)
                for h in horizons}
