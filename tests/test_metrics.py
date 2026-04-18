"""Unit tests for src/evaluation/metrics.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.evaluation.metrics import (
    rmse,
    skill_rmse,
    variance_retention,
    kge_components,
    skill_vp,
)


def test_rmse_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(y, y) == pytest.approx(0.0, abs=1e-10)


def test_rmse_known():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    # sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5), rel=1e-6)


def test_skill_rmse_baseline():
    """When model IS persistence, skill must be exactly 0."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pers = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    # model == persistence
    s = skill_rmse(y_true, y_pers, y_pers)
    assert s == pytest.approx(0.0, abs=1e-10)


def test_skill_rmse_perfect_model():
    """Perfect model achieves skill = 1."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pers = np.array([0.0, 0.0, 0.0])
    s = skill_rmse(y_true, y_true, y_pers)
    assert s == pytest.approx(1.0, abs=1e-10)


def test_skill_rmse_zero_persistence():
    """Returns NaN when persistence RMSE is zero (trivial series)."""
    y = np.array([5.0, 5.0, 5.0])
    result = skill_rmse(y, y, y)
    assert np.isnan(result)


def test_variance_retention_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert variance_retention(y, y) == pytest.approx(1.0, abs=1e-10)


def test_variance_retention_constant_pred():
    """Constant prediction means zero predicted variance → VR = 0."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full(4, 2.5)
    assert variance_retention(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)


def test_variance_retention_zero_true_variance():
    y = np.array([3.0, 3.0, 3.0])
    assert np.isnan(variance_retention(y, y))


def test_kge_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d = kge_components(y, y)
    assert d["kge"]   == pytest.approx(1.0, abs=1e-6)
    assert d["r"]     == pytest.approx(1.0, abs=1e-6)
    assert d["alpha"] == pytest.approx(1.0, abs=1e-6)
    assert d["beta"]  == pytest.approx(1.0, abs=1e-6)


def test_kge_bias_only():
    """Double the mean → beta=2, r=1, alpha=1 → kge = 1 - sqrt(0+0+1)."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true * 2.0
    d = kge_components(y_true, y_pred)
    assert d["beta"]  == pytest.approx(2.0, rel=1e-4)
    assert d["r"]     == pytest.approx(1.0, abs=1e-4)
    assert d["alpha"] == pytest.approx(1.0, abs=1e-4)
    # kge = 1 - sqrt((1-1)^2 + (1-1)^2 + (2-1)^2) = 1 - 1 = 0
    assert d["kge"]   == pytest.approx(0.0, abs=1e-4)


def test_kge_zero_mean_true():
    """Returns all NaN when mean(y_true) == 0 (undefined beta)."""
    y_true = np.zeros(5)
    y_pred = np.ones(5)
    d = kge_components(y_true, y_pred)
    assert np.isnan(d["kge"])


def test_skill_vp_no_penalty():
    """When VR >= 1, Skill_VP equals Skill_RMSE."""
    sk = 0.35
    vr = 1.2
    assert skill_vp(sk, vr) == pytest.approx(sk, abs=1e-10)


def test_skill_vp_at_boundary():
    """When VR == 1 exactly, no penalty is applied."""
    sk = 0.5
    assert skill_vp(sk, 1.0) == pytest.approx(sk, abs=1e-10)


def test_skill_vp_with_penalty():
    """Skill_VP = 0.2 * min(1, 0.5) = 0.1."""
    assert skill_vp(0.2, 0.5) == pytest.approx(0.1, abs=1e-10)


def test_skill_vp_full_collapse():
    """VR = 0 (constant prediction) → Skill_VP = 0 regardless of skill."""
    assert skill_vp(0.8, 0.0) == pytest.approx(0.0, abs=1e-10)
