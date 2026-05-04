"""Paper C — LightGBM capacity-sweep summary builder.

Reads the metrics produced by ``run_paper_c_lightgbm_capacity_sweep.py``
and generates:

    * results/paper_c_lightgbm_capacity_summary.csv
        Compact comparison table by station x horizon x configuration,
        restricted to the columns most relevant for the paper:
        rmse_model, skill_rmse, kge_alpha, vr, skill_vp.

    * results/paper_c_lightgbm_capacity_findings.txt
        Plain-text, automatically generated reading of the table:
        per-station best-RMSE / lowest-alpha configuration per horizon,
        global counts of how often ``lgbm_large`` beats ``lgbm_small``
        on RMSE and on alpha, plus a sober trade-off note.

    * results/fig_paper_c_lgbm_capacity_alpha_rmse.png
        Single matplotlib figure (no seaborn, no custom colors).

Usage:
    python3 experiments/build_paper_c_lightgbm_capacity_summary.py
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


METRICS_PATH = ROOT / "results" / "metrics_lightgbm_capacity_sweep.csv"
SUMMARY_PATH = ROOT / "results" / "paper_c_lightgbm_capacity_summary.csv"
FINDINGS_PATH = ROOT / "results" / "paper_c_lightgbm_capacity_findings.txt"
FIG_PATH = ROOT / "results" / "fig_paper_c_lgbm_capacity_alpha_rmse.png"

CONFIG_ORDER = ["lgbm_small", "lgbm_base", "lgbm_large"]
SUMMARY_COLUMNS = [
    "station",
    "horizon",
    "model",
    "rmse_model",
    "skill_rmse",
    "kge_alpha",
    "vr",
    "skill_vp",
]


def _build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in SUMMARY_COLUMNS if c not in metrics.columns]
    if missing:
        raise ValueError(f"Input metrics CSV is missing columns: {missing}")
    summary = metrics[SUMMARY_COLUMNS].copy()
    model_cat = pd.CategoricalDtype(categories=CONFIG_ORDER, ordered=True)
    summary["model"] = summary["model"].astype(model_cat)
    summary = summary.sort_values(["station", "horizon", "model"]).reset_index(drop=True)
    return summary


def _argmin_safe(series: pd.Series) -> str | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return str(valid.idxmin())


def _build_findings(summary: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("Paper C - LightGBM capacity-sweep findings")
    lines.append("=" * 60)
    lines.append("")
    lines.append("All numbers are computed directly from")
    lines.append("results/metrics_lightgbm_capacity_sweep.csv — no values are hardcoded.")
    lines.append("")

    stations = sorted(summary["station"].unique())
    horizons = sorted(summary["horizon"].unique())

    # Per-station, per-horizon: which configuration minimises RMSE / alpha.
    for station in stations:
        lines.append(f"## Station: {station}")
        sub = summary[summary["station"] == station]
        for h in horizons:
            row = sub[sub["horizon"] == h].set_index("model")
            if row.empty:
                continue

            rmse_winner = _argmin_safe(row["rmse_model"])
            # "Lowest alpha" follows the user's brief literally; we also
            # report the |alpha-1| view because that is what variance
            # collapse actually means.
            alpha_winner = _argmin_safe(row["kge_alpha"])
            alpha_dev = (row["kge_alpha"] - 1.0).abs()
            alpha_closest = _argmin_safe(alpha_dev)

            rmse_str = "n/a" if rmse_winner is None else (
                f"{rmse_winner} (RMSE={row.loc[rmse_winner, 'rmse_model']:.3f})"
            )
            alpha_low_str = "n/a" if alpha_winner is None else (
                f"{alpha_winner} (alpha={row.loc[alpha_winner, 'kge_alpha']:.3f})"
            )
            alpha_close_str = "n/a" if alpha_closest is None else (
                f"{alpha_closest} (alpha={row.loc[alpha_closest, 'kge_alpha']:.3f})"
            )

            lines.append(
                f"  h={h:>2}h | best RMSE: {rmse_str:<40} "
                f"| lowest alpha: {alpha_low_str:<35} "
                f"| alpha closest to 1: {alpha_close_str}"
            )
        lines.append("")

    # Global counts: large vs small.
    pivot_rmse = summary.pivot_table(
        index=["station", "horizon"], columns="model", values="rmse_model", observed=False
    )
    pivot_alpha = summary.pivot_table(
        index=["station", "horizon"], columns="model", values="kge_alpha", observed=False
    )

    n_combos = len(pivot_rmse)
    if {"lgbm_large", "lgbm_small"}.issubset(pivot_rmse.columns):
        large_better_rmse = int(
            (pivot_rmse["lgbm_large"] < pivot_rmse["lgbm_small"]).fillna(False).sum()
        )
    else:
        large_better_rmse = 0
    if {"lgbm_large", "lgbm_small"}.issubset(pivot_alpha.columns):
        large_lower_alpha = int(
            (pivot_alpha["lgbm_large"] < pivot_alpha["lgbm_small"]).fillna(False).sum()
        )
    else:
        large_lower_alpha = 0

    lines.append("## Global counts (across station x horizon combinations)")
    lines.append(f"Total combinations evaluated: {n_combos}")
    lines.append(
        f"lgbm_large < lgbm_small on RMSE      : {large_better_rmse} / {n_combos}"
    )
    lines.append(
        f"lgbm_large < lgbm_small on kge_alpha : {large_lower_alpha} / {n_combos}"
    )
    lines.append("")

    # Trade-off note (sober, in academic English, no overclaim).
    lines.append("## Reading")
    if n_combos == 0:
        lines.append(
            "No valid combinations were found in the input CSV; no reading produced."
        )
    else:
        rmse_share = large_better_rmse / n_combos
        alpha_share = large_lower_alpha / n_combos
        if rmse_share >= 0.5 and alpha_share <= 0.5:
            tradeoff = (
                "Increasing model capacity tends to reduce point-error (RMSE) but "
                "does not consistently reduce kge_alpha; in several configurations "
                "the larger model retains or amplifies the variance-amplitude shift. "
                "This is consistent with a trade-off between point-error minimisation "
                "and structural fidelity along the capacity axis."
            )
        elif rmse_share <= 0.5 and alpha_share >= 0.5:
            tradeoff = (
                "Larger capacity does not yield consistent RMSE gains in this setting, "
                "yet it tends to produce lower kge_alpha values. Whether this should be "
                "read as improved or degraded variance fidelity depends on whether the "
                "base model already over- or under-disperses."
            )
        elif rmse_share >= 0.5 and alpha_share >= 0.5:
            tradeoff = (
                "Larger capacity tends to improve RMSE while also lowering kge_alpha. "
                "No clean trade-off emerges along the capacity axis in this experiment."
            )
        else:
            tradeoff = (
                "Neither RMSE nor kge_alpha moves consistently with capacity in this "
                "experiment, suggesting that capacity alone is not the dominant driver "
                "of the RMSE / structural-fidelity decoupling reported in Paper C."
            )
        lines.append(tradeoff)
    lines.append("")

    return "\n".join(lines)


def _build_figure(summary: pd.DataFrame) -> None:
    stations = sorted(summary["station"].unique())
    horizons = sorted(summary["horizon"].unique())
    n_stations = len(stations)
    if n_stations == 0:
        return

    fig, axes = plt.subplots(
        n_stations,
        2,
        figsize=(10, 3.2 * n_stations),
        squeeze=False,
        sharex=True,
    )

    x = np.arange(len(horizons))
    width = 0.27

    for row, station in enumerate(stations):
        sub = summary[summary["station"] == station]
        ax_rmse, ax_alpha = axes[row, 0], axes[row, 1]

        for i, cfg in enumerate(CONFIG_ORDER):
            cfg_rows = sub[sub["model"] == cfg].set_index("horizon")
            rmse_vals = [cfg_rows.loc[h, "rmse_model"] if h in cfg_rows.index else np.nan
                         for h in horizons]
            alpha_vals = [cfg_rows.loc[h, "kge_alpha"] if h in cfg_rows.index else np.nan
                          for h in horizons]
            ax_rmse.bar(x + (i - 1) * width, rmse_vals, width=width, label=cfg)
            ax_alpha.bar(x + (i - 1) * width, alpha_vals, width=width, label=cfg)

        ax_rmse.set_title(f"{station} — RMSE by horizon")
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.set_xticks(x)
        ax_rmse.set_xticklabels([f"h={h}" for h in horizons])

        ax_alpha.set_title(f"{station} — KGE alpha by horizon")
        ax_alpha.set_ylabel("kge_alpha")
        ax_alpha.axhline(1.0, linestyle="--", linewidth=0.8)
        ax_alpha.set_xticks(x)
        ax_alpha.set_xticklabels([f"h={h}" for h in horizons])

        if row == 0:
            ax_rmse.legend(loc="best", fontsize=8)
            ax_alpha.legend(loc="best", fontsize=8)

    fig.suptitle("Paper C — LightGBM capacity sweep: RMSE vs KGE alpha")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    if not METRICS_PATH.exists():
        print(f"Error: input metrics not found at {METRICS_PATH}")
        print("Run experiments/run_paper_c_lightgbm_capacity_sweep.py first.")
        return

    metrics = pd.read_csv(METRICS_PATH)
    summary = _build_summary(metrics)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Summary saved to {SUMMARY_PATH}")

    findings = _build_findings(summary)
    FINDINGS_PATH.write_text(findings, encoding="utf-8")
    print(f"Findings saved to {FINDINGS_PATH}")

    try:
        _build_figure(summary)
        print(f"Figure saved to {FIG_PATH}")
    except Exception as exc:  # figure is optional; never block the summary
        print(f"[WARN] Could not build figure: {exc}")


if __name__ == "__main__":
    main()
