"""Unit tests for src/rolling_origin.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.rolling_origin import generate_folds


def _make_series(n: int = 12000) -> pd.Series:
    idx = pd.date_range("2019-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    return pd.Series(rng.uniform(0, 100, n), index=idx, name="pm10")


def test_fold_count():
    """Exactly n_folds folds must be generated."""
    series = _make_series(12000)
    folds = list(generate_folds(series, n_folds=16, min_train_size=8760))
    assert len(folds) == 16


def test_min_train_size():
    """First fold must contain at least min_train_size observations."""
    series = _make_series(12000)
    first_fold = next(generate_folds(series, n_folds=16, min_train_size=8760))
    assert len(first_fold["train_idx"]) >= 8760


def test_no_leakage():
    """No test index may fall inside the training window of the same fold."""
    series = _make_series(12000)
    for fold in generate_folds(series, n_folds=16, min_train_size=8760):
        train_set = set(fold["train_idx"].tolist())
        for h, positions in fold["test_idx"].items():
            for pos in positions:
                assert pos not in train_set, (
                    f"Leakage detected: fold {fold['fold']}, horizon {h}, "
                    f"test position {pos} is inside training set."
                )


def test_test_indices_after_train():
    """Every test index must strictly follow the last training index."""
    series = _make_series(12000)
    for fold in generate_folds(series, n_folds=16, min_train_size=8760):
        last_train = fold["train_idx"][-1]
        for h, positions in fold["test_idx"].items():
            for pos in positions:
                assert pos > last_train, (
                    f"Fold {fold['fold']}, h={h}: test pos {pos} "
                    f"<= last train pos {last_train}."
                )


def test_progressive_expansion():
    """Each successive fold's training window must be >= the previous one."""
    series = _make_series(12000)
    prev_len = 0
    for fold in generate_folds(series, n_folds=16, min_train_size=8760):
        cur_len = len(fold["train_idx"])
        assert cur_len >= prev_len
        prev_len = cur_len


def test_series_too_short():
    """ValueError must be raised when series is too short."""
    short = _make_series(100)
    with pytest.raises(ValueError, match="too short"):
        list(generate_folds(short, n_folds=16, min_train_size=8760))


def test_horizons_subset():
    """All requested horizons must appear as keys in test_idx."""
    series = _make_series(12000)
    horizons = [1, 6, 12, 24]
    for fold in generate_folds(series, n_folds=4, min_train_size=8760, horizons=horizons):
        assert set(fold["test_idx"].keys()) == set(horizons)
