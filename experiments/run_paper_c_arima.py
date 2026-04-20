"""Rolling-origin ARIMA baseline for Paper C.

This script mirrors the protocol used by ``run_paper_c_baseline.py`` so
the ARIMA outputs are directly comparable to the persistence baseline:

* Univariate PM10 only.
* One ARIMA model per station and per fold, refit from scratch on the
  training window of the fold (no data leakage).
* Rolling-origin folds identical to the persistence baseline
  (``n_folds=8``, ``min_train_size=2160``).
* Forecast horizons kept at ``[1, 6, 12, 24]``.
* Persistence predictions are still computed for every fold so skill
  scores are defined the same way for both baselines.

ARIMA configuration
-------------------
A fixed, defendable starter: ``ARIMA(2, 1, 2)`` (non-seasonal). The
first-order integration removes the non-stationary component while the
AR(2)/MA(2) terms capture short-range autocorrelation. This is
intentionally simple and does not rely on auto_arima or any
hyperparameter search, trading a bit of sophistication for stability.
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.rolling_origin import generate_folds
from src.models.arima_model import ARIMAModel
from src.evaluation.metrics import horizon_profile


ARIMA_ORDER = (2, 1, 2)


def run_arima_experiment():
    data_path = ROOT / "data" / "processed" / "pm10_two_station_hourly.csv"
    if not data_path.exists():
        print(f"Error: Processed data not found at {data_path}")
        return

    df = pd.read_csv(data_path, parse_dates=["timestamp"])

    horizons = [1, 6, 12, 24]
    n_folds = 8
    min_train_size = 2160

    all_metrics = []
    all_preds = []

    for station, group in df.groupby("station"):
        print(f"\nProcessing station: {station} (ARIMA{ARIMA_ORDER})")
        group = group.sort_values("timestamp").reset_index(drop=True)
        series = group["pm10"]
        values = series.values.astype(float)
        timestamps = group["timestamp"]

        accum = {h: {"y_true": [], "y_pred": [], "y_pers": [], "timestamp": []}
                 for h in horizons}

        for fold in generate_folds(series, n_folds=n_folds,
                                   min_train_size=min_train_size,
                                   horizons=horizons):
            origin_pos = fold["origin_pos"]
            origin_val = values[origin_pos]
            test_idx = fold["test_idx"]
            train_slice = fold["train_slice"]

            train_series = pd.Series(
                values[train_slice],
                index=timestamps.iloc[train_slice].values,
            )

            model = ARIMAModel(order=ARIMA_ORDER)
            try:
                preds = model.predict_fold(train_series, horizons)
            except Exception as exc:
                warnings.warn(
                    f"ARIMA fit failed for station={station} "
                    f"fold={fold['fold']}: {exc}. Falling back to persistence."
                )
                preds = {h: origin_val for h in horizons}

            for h in horizons:
                positions = test_idx[h]
                if len(positions) == 0:
                    continue
                pos = positions[0]

                y_true_val = values[pos]
                ts = group.loc[pos, "timestamp"]

                accum[h]["y_true"].append(y_true_val)
                accum[h]["y_pred"].append(preds[h])
                accum[h]["y_pers"].append(origin_val)
                accum[h]["timestamp"].append(ts)

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
                "prediction_arima": accum[h]["y_pred"],
                "prediction_persistence": accum[h]["y_pers"],
            })
            all_preds.append(h_df)

    if all_metrics:
        results_dir = ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        metrics_df = pd.concat(all_metrics, ignore_index=True)
        metrics_out = results_dir / "metrics_arima_by_horizon.csv"
        metrics_df.to_csv(metrics_out, index=False)
        print(f"Metrics saved to {metrics_out}")

        preds_df = pd.concat(all_preds, ignore_index=True)
        preds_out = results_dir / "predictions_arima.csv"
        preds_df.to_csv(preds_out, index=False)
        print(f"Predictions saved to {preds_out}")

        print("\n--- Summary ---")
        print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    run_arima_experiment()
