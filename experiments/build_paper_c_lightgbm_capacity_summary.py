"""Build summary table, findings text, and comparison figure for the LightGBM
capacity robustness sweep (Paper C).

Reads
-----
results/metrics_lightgbm_capacity_sweep.csv

Writes
------
results/paper_c_lightgbm_capacity_summary.csv
results/paper_c_lightgbm_capacity_findings.txt
results/fig_paper_c_lgbm_capacity_alpha_rmse.png

Usage
-----
    python experiments/build_paper_c_lightgbm_capacity_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = ROOT / "results"
SWEEP_CSV = RESULTS_DIR / "metrics_lightgbm_capacity_sweep.csv"
SUMMARY_CSV = RESULTS_DIR / "paper_c_lightgbm_capacity_summary.csv"
FINDINGS_TXT = RESULTS_DIR / "paper_c_lightgbm_capacity_findings.txt"
FIGURE_PNG = RESULTS_DIR / "fig_paper_c_lgbm_capacity_alpha_rmse.png"

MODEL_ORDER = ["lgbm_small", "lgbm_base", "lgbm_large"]
FOCUS_COLS = ["rmse_model", "skill_rmse", "kge_alpha", "vr", "skill_vp"]


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["station", "model", "horizon"] + FOCUS_COLS
    summary = df[keep].copy()
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    summary = summary.sort_values(["station", "horizon", "model"]).reset_index(drop=True)
    return summary


def build_findings(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("LightGBM Capacity Robustness Sweep — Paper C Findings")
    lines.append("=" * 60)
    lines.append("")

    stations = sorted(df["station"].unique())
    horizons = sorted(df["horizon"].unique())

    # Per-station, per-horizon best configuration
    for station in stations:
        lines.append(f"Station: {station}")
        lines.append("-" * 40)

        sdf = df[df["station"] == station]

        lines.append("  Best config by lowest RMSE per horizon:")
        for h in horizons:
            hdf = sdf[sdf["horizon"] == h]
            best_row = hdf.loc[hdf["rmse_model"].idxmin()]
            lines.append(
                f"    h={h:>2}h  ->  {best_row['model']}  "
                f"(RMSE={best_row['rmse_model']:.3f})"
            )

        lines.append("  Config with lowest kge_alpha per horizon:")
        for h in horizons:
            hdf = sdf[sdf["horizon"] == h]
            best_row = hdf.loc[hdf["kge_alpha"].idxmin()]
            lines.append(
                f"    h={h:>2}h  ->  {best_row['model']}  "
                f"(alpha={best_row['kge_alpha']:.3f})"
            )
        lines.append("")

    # Global counts
    lines.append("Global Counts (lgbm_large vs. lgbm_small)")
    lines.append("-" * 40)

    combos = [(s, h) for s in stations for h in horizons]
    n_total = len(combos)

    n_large_better_rmse = 0
    n_large_lower_alpha = 0

    for station, h in combos:
        sdf = df[(df["station"] == station) & (df["horizon"] == h)]
        small_row = sdf[sdf["model"] == "lgbm_small"]
        large_row = sdf[sdf["model"] == "lgbm_large"]
        if small_row.empty or large_row.empty:
            continue
        rmse_small = small_row["rmse_model"].values[0]
        rmse_large = large_row["rmse_model"].values[0]
        alpha_small = small_row["kge_alpha"].values[0]
        alpha_large = large_row["kge_alpha"].values[0]
        if rmse_large < rmse_small:
            n_large_better_rmse += 1
        if alpha_large < alpha_small:
            n_large_lower_alpha += 1

    lines.append(
        f"  lgbm_large lower RMSE than lgbm_small: "
        f"{n_large_better_rmse}/{n_total} station-horizon combinations"
    )
    lines.append(
        f"  lgbm_large lower kge_alpha than lgbm_small: "
        f"{n_large_lower_alpha}/{n_total} station-horizon combinations"
    )
    lines.append("")

    # Trade-off assessment
    lines.append("Trade-off Assessment")
    lines.append("-" * 40)

    rmse_improves = n_large_better_rmse > n_total / 2
    alpha_worsens = n_large_lower_alpha < n_total / 2  # lower alpha = more collapse

    if rmse_improves and not alpha_worsens:
        assessment = (
            "Increasing model capacity (lgbm_large vs. lgbm_small) tends to improve "
            "RMSE without consistently reducing variability fidelity (kge_alpha), "
            "suggesting no clear capacity-induced trade-off between point accuracy "
            "and structural fidelity in this dataset."
        )
    elif rmse_improves and alpha_worsens:
        assessment = (
            "Increasing model capacity (lgbm_large vs. lgbm_small) tends to improve "
            "RMSE but is associated with reduced variability fidelity (lower kge_alpha) "
            "in the majority of station-horizon combinations, indicating a capacity-induced "
            "trade-off between point accuracy and structural fidelity."
        )
    elif not rmse_improves and alpha_worsens:
        assessment = (
            "Increasing model capacity (lgbm_large vs. lgbm_small) does not consistently "
            "improve RMSE and is associated with reduced variability fidelity (lower "
            "kge_alpha), suggesting that larger capacity may increase variance collapse "
            "without compensating accuracy gains in this dataset."
        )
    else:
        assessment = (
            "No consistent directional pattern was observed between capacity level and "
            "either RMSE or variability fidelity (kge_alpha) across station-horizon "
            "combinations.  Results appear configuration- and horizon-dependent."
        )

    lines.append(f"  {assessment}")
    lines.append("")
    lines.append(
        "Note: All findings are derived directly from rolling-origin out-of-sample "
        "predictions.  No tuning was performed; configurations are fixed a priori."
    )

    return "\n".join(lines)


def build_figure(df: pd.DataFrame) -> None:
    stations = sorted(df["station"].unique())
    horizons = sorted(df["horizon"].unique())
    n_stations = len(stations)

    fig, axes = plt.subplots(
        2, n_stations,
        figsize=(5 * n_stations, 8),
        sharey=False,
    )
    if n_stations == 1:
        axes = axes.reshape(2, 1)

    x = np.arange(len(horizons))
    width = 0.25
    offsets = [-width, 0, width]
    hatches = ["", "//", "xx"]
    labels = MODEL_ORDER

    for col, station in enumerate(stations):
        sdf = df[df["station"] == station]

        for row, (metric, ylabel) in enumerate([
            ("rmse_model", "RMSE"),
            ("kge_alpha", "KGE alpha"),
        ]):
            ax = axes[row, col]
            for i, (model, offset, hatch) in enumerate(zip(MODEL_ORDER, offsets, hatches)):
                mdf = sdf[sdf["model"] == model].sort_values("horizon")
                values = mdf[metric].values
                ax.bar(
                    x + offset,
                    values,
                    width=width,
                    label=model if col == 0 and row == 0 else None,
                    hatch=hatch,
                    edgecolor="black",
                    linewidth=0.7,
                )

            ax.set_title(f"{station}" if row == 0 else "")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels([f"h={h}" for h in horizons])
            ax.set_xlabel("Horizon")
            ax.tick_params(axis="x", labelsize=8)

            if metric == "kge_alpha":
                ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    fig.legend(
        labels,
        loc="upper center",
        ncol=len(MODEL_ORDER),
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
    )
    fig.suptitle(
        "LightGBM Capacity Sweep — RMSE and KGE Alpha by Station and Horizon",
        y=1.04,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not SWEEP_CSV.exists():
        print(f"Error: {SWEEP_CSV} not found.")
        print("Run experiments/run_paper_c_lightgbm_capacity_sweep.py first.")
        return

    df = pd.read_csv(SWEEP_CSV)

    # Validate expected models are present
    present = set(df["model"].unique())
    missing = set(MODEL_ORDER) - present
    if missing:
        print(f"Warning: missing configurations in sweep CSV: {missing}")

    summary = build_summary(df)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved: {SUMMARY_CSV}")

    findings = build_findings(df)
    FINDINGS_TXT.write_text(findings, encoding="utf-8")
    print(f"Saved: {FINDINGS_TXT}")

    build_figure(df)
    print(f"Saved: {FIGURE_PNG}")

    print("\n--- Summary preview ---")
    print(summary.to_string(index=False))
    print("\n--- Findings ---")
    print(findings)


if __name__ == "__main__":
    main()
