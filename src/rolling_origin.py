"""Rolling-origin cross-validation fold generator.

Scheme
------
Given a series of T observations the folds are constructed as follows:

    fold k  (k = 0 … n_folds-1):
        train: [0, train_end_k)        length = min_train_size + k * step
        test:  for each horizon h,
               index  train_end_k + h - 1  (1-step-ahead origin)

where step = (T - min_train_size - max_horizon) // n_folds.

No test index ever falls inside the corresponding training window:
    min(test_indices_fold_k) = train_end_k + 1  (since h >= 1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterator


def generate_folds(
    series: pd.Series,
    n_folds: int = 16,
    min_train_size: int = 8760,
    horizons: list[int] | None = None,
) -> Iterator[dict]:
    """Yield rolling-origin fold descriptors.

    Each yielded dict has the structure::

        {
            'fold': int,                         # 0-indexed fold number
            'train_idx': np.ndarray,             # integer positions for training
            'test_idx': {h: np.ndarray},         # {horizon: positions array}
            'train_slice': slice,                # slice for .iloc
            'origin_pos': int,                   # last training position
        }

    Parameters
    ----------
    series:
        Full PM10 time series (DatetimeIndex, hourly).
    n_folds:
        Number of rolling-origin folds to generate.
    min_train_size:
        Minimum number of observations in the first training window
        (default 8760 ≈ one year of hourly data).
    horizons:
        List of forecast horizons in hours (default [1, 6, 12, 24]).

    Raises
    ------
    ValueError
        If the series is too short to accommodate the requested folds.
    """
    if horizons is None:
        horizons = [1, 6, 12, 24]

    T = len(series)
    max_horizon = max(horizons)

    required = min_train_size + max_horizon + n_folds
    if T < required:
        msg = (
            f"Configuration Conflict: series length (T={T}) is insufficient. "
            f"Requires at least {required} observations to satisfy constraints: "
            f"min_train_size={min_train_size}, max_horizon={max_horizon}, n_folds={n_folds}."
        )
        raise ValueError(msg)

    step = (T - min_train_size - max_horizon) // n_folds

    for k in range(n_folds):
        train_end = min_train_size + k * step  # exclusive upper bound
        train_idx = np.arange(0, train_end)
        origin_pos = train_end - 1  # last observed value

        test_idx: dict[int, np.ndarray] = {}
        for h in horizons:
            target_pos = train_end + h - 1  # index of the h-step-ahead target
            if target_pos < T:
                test_idx[h] = np.array([target_pos])
            else:
                test_idx[h] = np.array([], dtype=int)

        yield {
            "fold": k,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_slice": slice(0, train_end),
            "origin_pos": origin_pos,
        }


def collect_fold_predictions(
    series: pd.Series,
    model,
    n_folds: int = 16,
    min_train_size: int = 8760,
    horizons: list[int] | None = None,
) -> dict[int, dict]:
    """Convenience wrapper: run a model over all folds and collect arrays.

    Returns
    -------
    {horizon: {'y_true': np.ndarray, 'y_pred': np.ndarray}}
    """
    if horizons is None:
        horizons = [1, 6, 12, 24]

    accum: dict[int, dict[str, list]] = {h: {"y_true": [], "y_pred": []} for h in horizons}

    values = series.values.astype(float)

    for fold in generate_folds(series, n_folds=n_folds, min_train_size=min_train_size, horizons=horizons):
        train_vals = values[fold["train_idx"]]
        train_series = pd.Series(train_vals, index=series.index[fold["train_idx"]])
        model.fit(train_series)

        origin_val = values[fold["origin_pos"]]

        for h in horizons:
            test_positions = fold["test_idx"][h]
            if len(test_positions) == 0:
                continue
            y_true = values[test_positions[0]]
            y_pred = model.predict(origin_val, h)
            accum[h]["y_true"].append(y_true)
            accum[h]["y_pred"].append(y_pred)

    return {h: {"y_true": np.array(v["y_true"]), "y_pred": np.array(v["y_pred"])}
            for h, v in accum.items()}
