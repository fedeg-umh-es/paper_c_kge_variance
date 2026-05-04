"""Paper C — LightGBM capacity-sweep robustness experiment.

Runs three fixed LightGBM configurations (small / base / large) over the
exact Paper C rolling-origin protocol used by ``run_paper_c_baseline.py``.

Protocol (must match the rest of Paper C):
    * dataset:        data/processed/pm10_two_station_hourly.csv
    * stations:       Madrid_60, Valencia_Politecnico
    * horizons:       [1, 6, 12, 24]
    * folds:          rolling-origin, 8 folds, min_train_size=2160
    * preprocessing:  train-only (LightGBM is fit on the training window
                      only; no leakage)

Only the model capacity is varied:
    * lgbm_small : n_estimators=100,  num_leaves=15, max_depth=3
    * lgbm_base  : n_estimators=100,  num_leaves=31, max_depth=6
                   (= the existing ``LGBMModel`` defaults used by
                    ``experiments/run_lgbm.py``)
    * lgbm_large : n_estimators=500,  num_leaves=63, max_depth=7

Output:
    results/metrics_lightgbm_capacity_sweep.csv

Usage:
    python3 experiments/run_paper_c_lightgbm_capacity_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.rolling_origin import generate_folds
from src.models.lgbm_model import LGBMModel
from src.evaluation.metrics import horizon_profile


HORIZONS = [1, 6, 12, 24]
N_FOLDS = 8
MIN_TRAIN_SIZE = 2160
STATIONS = ["Madrid_60", "Valencia_Politecnico"]

# Three fixed capacity configurations. ``lgbm_base`` reproduces the
# defaults already used by ``src/models/lgbm_model.py`` (n_estimators=100,
# max_depth=6) and matches the configuration in ``experiments/run_lgbm.py``.
CAPACITY_CONFIGS = {
    "lgbm_small": {
        "n_estimators": 100,
        "num_leaves": 15,
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_jobs": 1,
        "random_state": 42,
    },
    "lgbm_base": {
        "n_estimators": 100,
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": None,   # keep LightGBM default (0.1)
        "n_jobs": 1,
        "random_state": 42,
    },
    "lgbm_large": {
        "n_estimators": 500,
        "num_leaves": 63,
        "max_depth": 7,
        "learning_rate": 0.05,
        "n_jobs": 1,
        "random_state": 42,
    },
}


def _run_one_configuration(
    series: pd.Series,
    cfg: dict,
) -> pd.DataFrame:
    """Execute the rolling-origin protocol for one LightGBM configuration."""
    values = series.values.astype(float)
    accum = {h: {"y_true": [], "y_pred": [], "y_pers": []} for h in HORIZONS}

    for fold in generate_folds(
        series,
        n_folds=N_FOLDS,
        min_train_size=MIN_TRAIN_SIZE,
        horizons=HORIZONS,
    ):
        train_idx = fold["train_idx"]
        origin_pos = fold["origin_pos"]
        test_idx = fold["test_idx"]

        train_series = series.iloc[train_idx]
        origin_val = values[origin_pos]

        model = LGBMModel(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            num_leaves=cfg["num_leaves"],
            learning_rate=cfg["learning_rate"],
            n_jobs=cfg["n_jobs"],
            random_state=cfg["random_state"],
            horizons=HORIZONS,
        )
        model.fit(train_series)

        context = values[max(0, origin_pos - model.n_lags + 1): origin_pos + 1]

        for h in HORIZONS:
            positions = test_idx[h]
            if len(positions) == 0:
                continue
            pos = positions[0]
            accum[h]["y_true"].append(values[pos])
            accum[h]["y_pred"].append(model.predict(context, h))
            accum[h]["y_pers"].append(origin_val)

    y_true_dict = {h: np.array(v["y_true"]) for h, v in accum.items()}
    y_pred_dict = {h: np.array(v["y_pred"]) for h, v in accum.items()}
    y_pers_dict = {h: np.array(v["y_pers"]) for h, v in accum.items()}

    return horizon_profile(y_true_dict, y_pred_dict, y_pers_dict, HORIZONS)


def run_capacity_sweep() -> pd.DataFrame | None:
    data_path = ROOT / "data" / "processed" / "pm10_two_station_hourly.csv"
    if not data_path.exists():
        print(f"Error: Processed data not found at {data_path}")
        print("Run src/preprocessing/build_unified_pm10_dataset.py first.")
        return None

    df = pd.read_csv(data_path, parse_dates=["timestamp"])

    all_metrics: list[pd.DataFrame] = []

    for station in STATIONS:
        group = df[df["station"] == station].sort_values("timestamp").reset_index(drop=True)
        if group.empty:
            print(f"[SKIP] No rows for station {station}")
            continue

        series = group["pm10"]
        print(f"\n=== Station: {station}  (n={len(series)}) ===")

        for cfg_name, cfg in CAPACITY_CONFIGS.items():
            print(f"  -> {cfg_name}")
            try:
                profile = _run_one_configuration(series, cfg)
            except ValueError as exc:
                print(f"     [SKIP] {exc}")
                continue
            profile.insert(0, "model", cfg_name)
            profile.insert(0, "station", station)
            all_metrics.append(profile)

    if not all_metrics:
        print("No results produced.")
        return None

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "metrics_lightgbm_capacity_sweep.csv"
    metrics_df.to_csv(out_path, index=False)
    print(f"\nMetrics saved to {out_path}")
    print("\n--- Summary ---")
    print(metrics_df.to_string(index=False))
    return metrics_df


if __name__ == "__main__":
    run_capacity_sweep()
