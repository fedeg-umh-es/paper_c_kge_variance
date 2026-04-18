"""Master experiment runner.

Executes all experiments in sequence and consolidates outputs.

Usage
-----
    python experiments/run_all.py
    python experiments/run_all.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

RESULTS_DIR = ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main(dry_run: bool = False) -> None:
    # Run each experiment module sequentially
    from experiments import run_baseline, run_lstm, run_lgbm

    print("=" * 60)
    print("Step 1/3  Baselines (Persistence + ARIMA)")
    print("=" * 60)
    run_baseline.main(dry_run=dry_run)

    print("\n" + "=" * 60)
    print("Step 2/3  LSTM")
    print("=" * 60)
    run_lstm.main(dry_run=dry_run)

    print("\n" + "=" * 60)
    print("Step 3/3  LightGBM")
    print("=" * 60)
    run_lgbm.main(dry_run=dry_run)

    if dry_run:
        print("\n[dry-run] All modules loaded successfully.")
        return

    # Consolidate individual result CSVs
    csv_files = [
        RESULTS_DIR / "baseline_results.csv",
        RESULTS_DIR / "lstm_results.csv",
        RESULTS_DIR / "lgbm_results.csv",
    ]
    frames = [pd.read_csv(f) for f in csv_files if f.exists()]
    if not frames:
        print("No results to consolidate.")
        return

    full = pd.concat(frames, ignore_index=True)
    full.to_csv(RESULTS_DIR / "full_comparison.csv", index=False)
    print(f"\nFull comparison saved to {RESULTS_DIR / 'full_comparison.csv'}")

    # Rank comparison
    from src.evaluation.diagnostics import rank_comparison
    results_dict = {
        name: grp.drop(columns=["station", "model"])
        for name, grp in full.groupby("model")
    }
    # Reset index for each sub-DataFrame
    results_dict = {
        name: df.groupby("horizon").mean().reset_index()
        for name, df in results_dict.items()
    }

    for metric in ["skill_rmse", "kge", "skill_vp"]:
        try:
            rank_df = rank_comparison(results_dict, metric=metric)
            rank_df.to_csv(RESULTS_DIR / f"rank_{metric}.csv")
        except Exception as exc:
            print(f"  [WARN] rank_comparison failed for {metric}: {exc}")

    print(f"Rank tables saved to {RESULTS_DIR}")

    # Generate figures
    try:
        from src.visualization.plots import (
            plot_horizon_metric,
            plot_kge_components,
            plot_variance_retention,
            plot_skill_vp,
        )
        plot_horizon_metric(results_dict, metric="skill_rmse",
                            ylabel="Skill$_{RMSE}$", filename="skill_rmse")
        plot_kge_components(results_dict)
        plot_variance_retention(results_dict)
        plot_skill_vp(results_dict)
        print("Figures saved to results/figures/")
    except Exception as exc:
        print(f"[WARN] Figure generation failed: {exc}")

    print("\nAll experiments complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all forecasting experiments.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
