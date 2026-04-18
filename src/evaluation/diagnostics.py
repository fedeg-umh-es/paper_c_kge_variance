"""Multi-model diagnostic tables for MATCOM paper."""

from __future__ import annotations

import pandas as pd
import numpy as np


def summary_table(results_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a multi-model summary table from per-model horizon profiles.

    Parameters
    ----------
    results_dict:
        {model_name: horizon_profile_df} where each DataFrame is the output
        of metrics.horizon_profile().

    Returns
    -------
    pd.DataFrame
        Multi-level columns (model, metric) indexed by horizon.
        Formatted for LaTeX booktabs export via .to_latex(booktabs=True).
    """
    frames = {}
    for model_name, df in results_dict.items():
        df = df.set_index("horizon")
        frames[model_name] = df

    combined = pd.concat(frames, axis=1)
    combined.index.name = "horizon"
    return combined


def rank_comparison(
    results_dict: dict[str, pd.DataFrame],
    metric: str = "skill_rmse",
) -> pd.DataFrame:
    """Rank models by a given metric for each forecast horizon.

    Higher values are assumed better for all metrics except 'rmse_model'
    and 'rmse_persistence' (where lower is better).

    Parameters
    ----------
    results_dict:
        {model_name: horizon_profile_df}.
    metric:
        Column name from horizon_profile DataFrame.

    Returns
    -------
    pd.DataFrame
        Rows = horizons, columns = model names, values = rank (1 = best).
    """
    lower_is_better = {"rmse_model", "rmse_persistence"}

    # Collect metric values per horizon
    horizon_vals: dict[int, dict[str, float]] = {}
    for model_name, df in results_dict.items():
        for _, row in df.iterrows():
            h = int(row["horizon"])
            if h not in horizon_vals:
                horizon_vals[h] = {}
            horizon_vals[h][model_name] = float(row.get(metric, np.nan))

    ranks = {}
    for h, model_scores in sorted(horizon_vals.items()):
        series = pd.Series(model_scores).dropna()
        if metric in lower_is_better:
            ranked = series.rank(ascending=True)
        else:
            ranked = series.rank(ascending=False)
        ranks[h] = ranked

    rank_df = pd.DataFrame(ranks).T
    rank_df.index.name = "horizon"
    return rank_df


def to_latex_booktabs(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    """Return a booktabs-compatible LaTeX table string."""
    return df.to_latex(
        float_format=lambda x: float_fmt.format(x) if pd.notna(x) else "--",
        na_rep="--",
        bold_rows=False,
        multicolumn=True,
        multicolumn_format="c",
    )
