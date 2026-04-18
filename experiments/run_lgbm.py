"""LightGBM experiment.

Usage
-----
    python experiments/run_lgbm.py
    python experiments/run_lgbm.py --dry-run
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
from src.rolling_origin import generate_folds
from src.models.lgbm_model import LGBMModel
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
        print("[dry-run] LightGBM imports and config OK.")
        return

    horizons = cfg["horizons"]
    n_folds = cfg["n_folds"]
    min_train = cfg["min_train_size"]

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
            origin_val = values[origin_pos]

            lgbm = LGBMModel(horizons=horizons)
            lgbm.fit(train_s)

            context = values[max(0, origin_pos - lgbm.n_lags + 1): origin_pos + 1]

            for h in horizons:
                positions = test_idx[h]
                if len(positions) == 0:
                    continue
                pos = positions[0]
                y_true_val = values[pos]
                y_pred_val = lgbm.predict(context, h)
                y_pers_val = origin_val

                accum[h]["y_true"].append(y_true_val)
                accum[h]["y_pred"].append(y_pred_val)
                accum[h]["y_pers"].append(y_pers_val)

            if (k + 1) % 4 == 0:
                print(f"  Fold {k + 1}/{n_folds} done")

        y_true_dict = {h: np.array(v["y_true"]) for h, v in accum.items()}
        y_pred_dict = {h: np.array(v["y_pred"]) for h, v in accum.items()}
        y_pers_dict = {h: np.array(v["y_pers"]) for h, v in accum.items()}

        profile = horizon_profile(y_true_dict, y_pred_dict, y_pers_dict, horizons)
        profile.insert(0, "model", "lgbm")
        profile.insert(0, "station", station)
        all_results.append(profile)

    if not all_results:
        print("No results — no processed data found.")
        return

    results_df = pd.concat(all_results, ignore_index=True)
    out_path = RESULTS_DIR / "lgbm_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
