"""Baseline experiments: Persistence and ARIMA models.

Usage
-----
    python experiments/run_baseline.py
    python experiments/run_baseline.py --dry-run   # import check only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root or from experiments/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.data_loader import load_processed_data
from src.rolling_origin import generate_folds
from src.models.persistence import PersistenceModel
from src.models.arima_model import ARIMAModel
from src.evaluation.metrics import horizon_profile

RESULTS_DIR = ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    cfg_path = ROOT / "experiments" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def run_model_on_station(
    series: pd.Series,
    model_cls,
    model_kwargs: dict,
    cfg: dict,
) -> pd.DataFrame:
    """Run a single model on all folds and average metric profiles."""
    horizons = cfg["horizons"]
    n_folds = cfg["n_folds"]
    min_train = cfg["min_train_size"]

    accum: dict[int, dict[str, list]] = {
        h: {"y_true": [], "y_pred": [], "y_pers": []} for h in horizons
    }

    values = series.values.astype(float)
    pers_model = PersistenceModel()

    for fold in generate_folds(series, n_folds=n_folds,
                                min_train_size=min_train,
                                horizons=horizons):
        k = fold["fold"]
        train_idx = fold["train_idx"]
        origin_pos = fold["origin_pos"]
        test_idx = fold["test_idx"]

        train_series = series.iloc[train_idx]
        origin_val = values[origin_pos]

        if isinstance(model_cls(), PersistenceModel):
            preds_per_h = {h: origin_val for h in horizons}
        else:
            m = model_cls(**model_kwargs)
            preds_per_h = m.predict_fold(train_series, horizons=horizons)

        for h in horizons:
            positions = test_idx[h]
            if len(positions) == 0:
                continue
            pos = positions[0]
            y_true_val = values[pos]
            y_pred_val = preds_per_h[h]
            y_pers_val = origin_val

            accum[h]["y_true"].append(y_true_val)
            accum[h]["y_pred"].append(y_pred_val)
            accum[h]["y_pers"].append(y_pers_val)

        if (k + 1) % 4 == 0:
            print(f"  Fold {k + 1}/{n_folds} done")

    y_true_dict = {h: np.array(v["y_true"]) for h, v in accum.items()}
    y_pred_dict = {h: np.array(v["y_pred"]) for h, v in accum.items()}
    y_pers_dict = {h: np.array(v["y_pers"]) for h, v in accum.items()}

    return horizon_profile(y_true_dict, y_pred_dict, y_pers_dict, horizons)


def main(dry_run: bool = False) -> None:
    cfg = load_config()

    if dry_run:
        print("[dry-run] All imports and config loaded successfully.")
        return

    all_results: list[pd.DataFrame] = []

    for station in cfg["station_codes"]:
        print(f"\n=== Station: {station} ===")
        try:
            series = load_processed_data(station)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            continue

        for model_name, model_cls, model_kwargs in [
            ("persistence", PersistenceModel, {}),
            ("arima",       ARIMAModel,       {"order": (1, 1, 1)}),
        ]:
            print(f"  Running {model_name}...")
            profile = run_model_on_station(series, model_cls, model_kwargs, cfg)
            profile.insert(0, "model", model_name)
            profile.insert(0, "station", station)
            all_results.append(profile)

    if not all_results:
        print("\nNo results generated — no processed data found.")
        return

    results_df = pd.concat(all_results, ignore_index=True)
    out_path = RESULTS_DIR / "baseline_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    summary = (
        results_df.groupby(["model", "horizon"])
        [["skill_rmse", "vr", "kge", "skill_vp"]]
        .mean()
        .round(4)
    )
    print("\n--- Summary (averaged over stations) ---")
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline forecasting experiments.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check imports and config only; do not run experiments.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
