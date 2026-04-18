"""LSTM experiment.

Usage
-----
    python experiments/run_lstm.py
    python experiments/run_lstm.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.data_loader import load_processed_data
from src.preprocessing import scale_fold
from src.rolling_origin import generate_folds
from src.models.lstm_model import LSTMModel
from src.models.persistence import PersistenceModel
from src.evaluation.metrics import horizon_profile

RESULTS_DIR = ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open(ROOT / "experiments" / "config.yaml") as f:
        return yaml.safe_load(f)


def main(dry_run: bool = False) -> None:
    cfg = load_config()

    if dry_run:
        print("[dry-run] LSTM imports and config OK.")
        return

    horizons = cfg["horizons"]
    n_folds = cfg["n_folds"]
    min_train = cfg["min_train_size"]
    context_window = 24

    all_results: list[pd.DataFrame] = []

    for station in cfg["station_codes"]:
        print(f"\n=== Station: {station} ===")
        try:
            series = load_processed_data(station)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            continue

        values = series.values.astype(float)
        accum = {h: {"y_true": [], "y_pred": [], "y_pers": []} for h in horizons}

        for fold in generate_folds(series, n_folds=n_folds,
                                    min_train_size=min_train,
                                    horizons=horizons):
            k = fold["fold"]
            train_idx = fold["train_idx"]
            origin_pos = fold["origin_pos"]
            test_idx = fold["test_idx"]

            train_s = series.iloc[train_idx]
            # Use a small look-ahead window for test scaling alignment
            # (scale on train only — leakage-free)
            test_placeholder = series.iloc[origin_pos : origin_pos + max(horizons) + 1]
            train_sc, _, scaler = scale_fold(train_s, test_placeholder)

            lstm = LSTMModel(context_window=context_window, horizons=horizons)
            lstm.fit(train_sc, scaler=scaler)

            origin_val_raw = values[origin_pos]
            context_raw = values[max(0, origin_pos - context_window + 1): origin_pos + 1]
            # Scale context using train-fitted scaler
            import numpy as _np
            context_sc = scaler.transform(context_raw.reshape(-1, 1)).ravel()

            for h in horizons:
                positions = test_idx[h]
                if len(positions) == 0:
                    continue
                pos = positions[0]
                y_true_sc = float(scaler.transform([[values[pos]]])[0, 0])
                y_pred_sc = lstm.predict(context_sc, h)
                y_pers_sc = float(scaler.transform([[origin_val_raw]])[0, 0])

                accum[h]["y_true"].append(y_true_sc)
                accum[h]["y_pred"].append(y_pred_sc)
                accum[h]["y_pers"].append(y_pers_sc)

            if (k + 1) % 4 == 0:
                print(f"  Fold {k + 1}/{n_folds} done")

        y_true_dict = {h: np.array(v["y_true"]) for h, v in accum.items()}
        y_pred_dict = {h: np.array(v["y_pred"]) for h, v in accum.items()}
        y_pers_dict = {h: np.array(v["y_pers"]) for h, v in accum.items()}

        profile = horizon_profile(y_true_dict, y_pred_dict, y_pers_dict, horizons)
        profile.insert(0, "model", "lstm")
        profile.insert(0, "station", station)
        all_results.append(profile)

    if not all_results:
        print("No results — no processed data found.")
        return

    results_df = pd.concat(all_results, ignore_index=True)
    out_path = RESULTS_DIR / "lstm_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
