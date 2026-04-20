"""Rolling-origin LightGBM experiment for Paper C.

Mirrors the protocol of ``run_paper_c_baseline.py`` and
``run_paper_c_arima.py`` so results are directly comparable:

* Univariate PM10 only.
* One set of LightGBM models (one per horizon) trained per station and
  per rolling-origin fold, strictly on that fold's training window.
* Rolling-origin folds: ``n_folds=8``, ``min_train_size=2160``.
* Horizons: [1, 6, 12, 24] hours.
* Persistence predictions are recomputed per fold so that skill scores
  are defined consistently across all experiments.

On any per-fold, per-horizon training or prediction failure the script
falls back to the persistence value (last observed), emits a ``warnings``
warning, and continues — the full run is never aborted.

Feature set
-----------
Autoregressive lags of PM10: [1, 2, 3, 6, 12, 24].
All lag values are drawn from the training window only (no leakage).

LightGBM configuration
-----------------------
Fixed, conservative hyperparameters — see ``src/models/lightgbm_model.py``
for documentation. No grid search. No auto-ML.
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.rolling_origin import generate_folds
from src.models.lightgbm_model import LightGBMModel, LGBM_PARAMS
from src.evaluation.metrics import horizon_profile


def run_lightgbm_experiment():
    data_path = ROOT / "data" / "processed" / "pm10_two_station_hourly.csv"
    if not data_path.exists():
        print(f"Error: Processed data not found at {data_path}")
        return

    df = pd.read_csv(data_path, parse_dates=["timestamp"])

    horizons = [1, 6, 12, 24]
    n_folds = 8
    min_train_size = 2160

    fallback_count = 0
    all_metrics = []
    all_preds = []

    for station, group in df.groupby("station"):
        print(f"\nProcessing station: {station} (LightGBM)")
        group = group.sort_values("timestamp").reset_index(drop=True)
        series = group["pm10"]
        values = series.values.astype(float)

        accum = {h: {"y_true": [], "y_pred": [], "y_pers": [], "timestamp": []}
                 for h in horizons}

        for fold in generate_folds(series, n_folds=n_folds,
                                   min_train_size=min_train_size,
                                   horizons=horizons):
            fold_num = fold["fold"]
            origin_pos = fold["origin_pos"]
            origin_val = values[origin_pos]
            test_idx = fold["test_idx"]
            train_values = values[fold["train_slice"]]

            model = LightGBMModel(params=LGBM_PARAMS.copy(), horizons=horizons)
            try:
                preds = model.fit_predict_fold(train_values, horizons)
            except Exception as exc:
                warnings.warn(
                    f"LightGBM fold {fold_num} station={station} crashed "
                    f"entirely: {exc}. Falling back to persistence for all horizons."
                )
                preds = {h: None for h in horizons}

            for h in horizons:
                positions = test_idx[h]
                if len(positions) == 0:
                    continue
                pos = positions[0]

                y_pred = preds.get(h)
                if y_pred is None:
                    warnings.warn(
                        f"LightGBM station={station} fold={fold_num} h={h}: "
                        f"no prediction available, falling back to persistence."
                    )
                    y_pred = origin_val
                    fallback_count += 1

                accum[h]["y_true"].append(values[pos])
                accum[h]["y_pred"].append(y_pred)
                accum[h]["y_pers"].append(origin_val)
                accum[h]["timestamp"].append(group.loc[pos, "timestamp"])

        y_true_dict = {h: np.array(v["y_true"]) for h, v in accum.items()}
        y_pred_dict = {h: np.array(v["y_pred"]) for h, v in accum.items()}
        y_pers_dict = {h: np.array(v["y_pers"]) for h, v in accum.items()}

        profile = horizon_profile(y_true_dict, y_pred_dict, y_pers_dict, horizons)
        profile.insert(0, "station", station)
        all_metrics.append(profile)

        for h in horizons:
            h_df = pd.DataFrame({
                "station": station,
                "horizon": h,
                "timestamp": accum[h]["timestamp"],
                "y_true": accum[h]["y_true"],
                "prediction_lightgbm": accum[h]["y_pred"],
                "prediction_persistence": accum[h]["y_pers"],
            })
            all_preds.append(h_df)

    if fallback_count:
        print(f"\nWarning: {fallback_count} fold/horizon(s) fell back to persistence.")

    if all_metrics:
        results_dir = ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        metrics_df = pd.concat(all_metrics, ignore_index=True)
        metrics_out = results_dir / "metrics_lightgbm_by_horizon.csv"
        metrics_df.to_csv(metrics_out, index=False)
        print(f"Metrics saved to {metrics_out}")

        preds_df = pd.concat(all_preds, ignore_index=True)
        preds_out = results_dir / "predictions_lightgbm.csv"
        preds_df.to_csv(preds_out, index=False)
        print(f"Predictions saved to {preds_out}")

        print("\n--- Summary ---")
        print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    run_lightgbm_experiment()
