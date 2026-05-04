"""LightGBM capacity robustness sweep for Paper C.

Evaluates three fixed capacity configurations of LightGBM under the same
rolling-origin protocol used in the Paper C baseline experiment.  Only model
capacity varies; features, lags, dataset, and evaluation protocol are
identical across configurations.

Output
------
results/metrics_lightgbm_capacity_sweep.csv

Usage
-----
    python experiments/run_paper_c_lightgbm_capacity_sweep.py
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
from src.models.persistence import PersistenceModel
from src.evaluation.metrics import horizon_profile

DATA_PATH = ROOT / "data" / "processed" / "pm10_two_station_hourly.csv"
RESULTS_DIR = ROOT / "results"

HORIZONS = [1, 6, 12, 24]
N_FOLDS = 8
MIN_TRAIN_SIZE = 2160

# Three fixed capacity configurations.
# lgbm_base reproduces the default LGBMModel constructor (n_estimators=100,
# num_leaves=31, max_depth=6, learning_rate=0.1) — the configuration used
# in the existing Paper C LightGBM experiment.
CONFIGS = {
    "lgbm_small": dict(
        n_estimators=100,
        num_leaves=15,
        max_depth=3,
        learning_rate=0.05,
        n_jobs=1,
    ),
    "lgbm_base": dict(
        n_estimators=100,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=1,
    ),
    "lgbm_large": dict(
        n_estimators=500,
        num_leaves=63,
        max_depth=7,
        learning_rate=0.05,
        n_jobs=1,
    ),
}


def run_station(station: str, series: pd.Series, config_name: str, config: dict) -> pd.DataFrame:
    values = series.values.astype(float)
    accum = {h: {"y_true": [], "y_pred": [], "y_pers": []} for h in HORIZONS}

    for fold in generate_folds(series, n_folds=N_FOLDS, min_train_size=MIN_TRAIN_SIZE, horizons=HORIZONS):
        train_idx = fold["train_idx"]
        origin_pos = fold["origin_pos"]
        test_idx = fold["test_idx"]

        train_s = series.iloc[train_idx]
        origin_val = values[origin_pos]

        model = LGBMModel(horizons=HORIZONS, **config)
        model.fit(train_s)

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

    profile = horizon_profile(y_true_dict, y_pred_dict, y_pers_dict, HORIZONS)
    profile.insert(0, "model", config_name)
    profile.insert(0, "station", station)
    return profile


def main() -> None:
    if not DATA_PATH.exists():
        print(f"Error: dataset not found at {DATA_PATH}")
        print("Run src/preprocessing/build_unified_pm10_dataset.py first.")
        return

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[pd.DataFrame] = []

    for station, group in df.groupby("station"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        series = group.set_index("timestamp")["pm10"]

        for config_name, config in CONFIGS.items():
            print(f"Station={station}  config={config_name} ...", end=" ", flush=True)
            result = run_station(station, series, config_name, config)
            all_results.append(result)
            print("done")

    if not all_results:
        print("No results produced.")
        return

    out = pd.concat(all_results, ignore_index=True)

    # Canonical column order
    col_order = [
        "station", "model", "horizon",
        "rmse_model", "rmse_persistence",
        "skill_rmse", "vr",
        "kge", "kge_r", "kge_alpha", "kge_beta",
        "skill_vp",
    ]
    out = out[col_order]

    out_path = RESULTS_DIR / "metrics_lightgbm_capacity_sweep.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
