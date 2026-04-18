"""Evaluation metrics for multi-horizon PM10 forecasting.

All functions operate on 1-D numpy arrays.

KGE reference
-------------
Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009).
Decomposition of the mean squared error and NSE performance criteria:
Implications for improving hydrological modelling.
Journal of Hydrology, 377(1-2), 80-91.
https://doi.org/10.1016/j.jhydrol.2009.08.003
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def skill_rmse(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_persistence: np.ndarray,
) -> float:
    """Skill score relative to persistence baseline.

    skill = 1 - RMSE_model / RMSE_persistence

    Positive values indicate the model outperforms persistence.
    Returns NaN if RMSE_persistence is zero.
    """
    rmse_model = rmse(y_true, y_pred_model)
    rmse_pers = rmse(y_true, y_pred_persistence)
    if rmse_pers == 0.0:
        return float("nan")
    return float(1.0 - rmse_model / rmse_pers)


def variance_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Variance Retention ratio.

    VR = Var(y_pred) / Var(y_true)

    VR < 1 indicates variance collapse (underdispersion).
    Returns NaN if Var(y_true) is zero.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    var_true = np.var(y_true)
    if var_true == 0.0:
        return float("nan")
    return float(np.var(y_pred) / var_true)


def kge_components(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Kling-Gupta Efficiency and its three decomposed components.

    Components
    ----------
    r     : Pearson correlation coefficient
    alpha : std(y_pred) / std(y_true)  — variability component
    beta  : mean(y_pred) / mean(y_true) — bias component
    kge   : 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)

    Returns NaN for kge and affected components if std or mean of y_true
    is zero.

    Reference: Gupta et al. (2009), J. Hydrology.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    if std_true == 0.0 or mean_true == 0.0:
        return {"r": float("nan"), "alpha": float("nan"),
                "beta": float("nan"), "kge": float("nan")}

    # Pearson r
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    alpha = float(std_pred / std_true)
    beta = float(mean_pred / mean_true)
    kge = float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))

    return {"r": r, "alpha": alpha, "beta": beta, "kge": kge}


def skill_vp(skill_rmse_val: float, vr_val: float) -> float:
    """Variance-Penalised Skill diagnostic.

    Skill_VP = Skill_RMSE * min(1, VR)

    This is a *diagnostic* combination of existing metrics, not a new
    standalone metric.  It penalises skill scores when the model exhibits
    variance collapse (VR < 1) while leaving them unchanged when VR >= 1.
    """
    return float(skill_rmse_val * min(1.0, vr_val))


def horizon_profile(
    y_true_dict: dict[int, np.ndarray],
    y_pred_dict: dict[int, np.ndarray],
    y_pers_dict: dict[int, np.ndarray],
    horizons: list[int],
) -> pd.DataFrame:
    """Compute the full metric profile for all forecast horizons.

    Parameters
    ----------
    y_true_dict:
        {horizon: 1-D array of observed values}.
    y_pred_dict:
        {horizon: 1-D array of model predictions}.
    y_pers_dict:
        {horizon: 1-D array of persistence predictions}.
    horizons:
        Ordered list of forecast horizons (e.g. [1, 6, 12, 24]).

    Returns
    -------
    pd.DataFrame with columns:
        horizon, rmse_model, rmse_persistence, skill_rmse,
        vr, kge, kge_r, kge_alpha, kge_beta, skill_vp
    """
    rows = []
    for h in horizons:
        yt = y_true_dict[h]
        yp = y_pred_dict[h]
        ype = y_pers_dict[h]

        if len(yt) == 0:
            continue

        rmse_m = rmse(yt, yp)
        rmse_p = rmse(yt, ype)
        sk = skill_rmse(yt, yp, ype)
        vr = variance_ratio(yt, yp)
        kge_d = kge_components(yt, yp)
        svp = skill_vp(sk, vr) if not (np.isnan(sk) or np.isnan(vr)) else float("nan")

        rows.append(
            {
                "horizon": h,
                "rmse_model": rmse_m,
                "rmse_persistence": rmse_p,
                "skill_rmse": sk,
                "vr": vr,
                "kge": kge_d["kge"],
                "kge_r": kge_d["r"],
                "kge_alpha": kge_d["alpha"],
                "kge_beta": kge_d["beta"],
                "skill_vp": svp,
            }
        )

    return pd.DataFrame(rows)
