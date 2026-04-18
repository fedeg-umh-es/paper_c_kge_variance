"""Train-only preprocessing utilities.

ABSOLUTE RULE: No transformation may use future information.
Every scaler, imputer, or encoder is fitted ONLY on the training split of
each rolling-origin fold, never on the test split or the full series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def clean_series(series: pd.Series) -> tuple[pd.Series, dict]:
    """Clean a raw hourly PM10 time series.

    Steps
    -----
    1. Drop duplicate timestamps (keep first).
    2. Reindex to a full hourly DatetimeIndex.
    3. Identify runs of consecutive NaN gaps > 6 h and record them as
       metadata — these are NOT imputed.

    Parameters
    ----------
    series:
        Raw PM10 values with a DatetimeIndex (any timezone).

    Returns
    -------
    (cleaned_series, metadata)
        cleaned_series: hourly Series; gaps > 6 h remain NaN.
        metadata: dict with keys
            'n_total', 'n_missing', 'coverage_pct', 'long_gaps' (list of
            (start, end, length_h) tuples for gaps > 6 h).
    """
    series = series.copy()
    series = series[~series.index.duplicated(keep="first")]
    series = series.sort_index()

    freq = "h"
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    series = series.reindex(full_idx)

    # Identify long gaps (> 6 consecutive NaN hours)
    is_nan = series.isna()
    long_gaps: list[tuple] = []
    in_gap = False
    gap_start = None
    gap_len = 0

    for ts, nan_flag in is_nan.items():
        if nan_flag:
            if not in_gap:
                in_gap = True
                gap_start = ts
                gap_len = 1
            else:
                gap_len += 1
        else:
            if in_gap and gap_len > 6:
                long_gaps.append((gap_start, ts, gap_len))
            in_gap = False
            gap_len = 0

    if in_gap and gap_len > 6:
        long_gaps.append((gap_start, series.index[-1], gap_len))

    n_total = len(series)
    n_missing = int(series.isna().sum())
    coverage = (n_total - n_missing) / n_total * 100 if n_total else 0.0

    metadata = {
        "n_total": n_total,
        "n_missing": n_missing,
        "coverage_pct": round(coverage, 2),
        "long_gaps": long_gaps,
    }
    return series, metadata


def scale_fold(
    train_series: pd.Series,
    test_series: pd.Series,
) -> tuple[pd.Series, pd.Series, MinMaxScaler]:
    """Fit MinMaxScaler on train only, apply to both splits.

    The scaler is fitted exclusively on `train_series`.  It is then applied
    to `test_series` without re-fitting — this enforces the leakage-free
    protocol.

    Parameters
    ----------
    train_series, test_series:
        1-D pandas Series of PM10 values.

    Returns
    -------
    (train_scaled, test_scaled, fitted_scaler)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_vals = train_series.values.reshape(-1, 1)
    scaler.fit(train_vals)

    train_scaled = pd.Series(
        scaler.transform(train_vals).ravel(),
        index=train_series.index,
        name=train_series.name,
    )
    test_scaled = pd.Series(
        scaler.transform(test_series.values.reshape(-1, 1)).ravel(),
        index=test_series.index,
        name=test_series.name,
    )
    return train_scaled, test_scaled, scaler
