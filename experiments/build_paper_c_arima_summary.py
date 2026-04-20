"""Build the Paper C ARIMA-vs-persistence summary table.

Lightweight post-processing script: loads the two existing metric CSVs
produced by ``run_paper_c_baseline.py`` and ``run_paper_c_arima.py``,
merges them by ``(station, horizon)``, adds a few derived diagnostic
columns, and writes a summary CSV plus a short human-readable findings
file.

Inputs
------
- results/metrics_baseline_by_horizon.csv
- results/metrics_arima_by_horizon.csv

Outputs
-------
- results/paper_c_arima_vs_persistence_summary.csv
- results/paper_c_arima_key_findings.txt

Only ``pandas`` and ``pathlib`` are used. No models are retrained.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

BASELINE_CSV = RESULTS_DIR / "metrics_baseline_by_horizon.csv"
ARIMA_CSV = RESULTS_DIR / "metrics_arima_by_horizon.csv"

SUMMARY_CSV = RESULTS_DIR / "paper_c_arima_vs_persistence_summary.csv"
FINDINGS_TXT = RESULTS_DIR / "paper_c_arima_key_findings.txt"

KEYS = ["station", "horizon"]

VARIANCE_COLLAPSE_THRESHOLD = 0.95
SEVERE_VARIANCE_COLLAPSE_THRESHOLD = 0.85
OVERDISPERSION_THRESHOLD = 1.05


def _rename_with_suffix(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Rename metric columns with an unambiguous suffix.

    The rmse_model column is renamed to ``rmse_<suffix>`` (so baseline
    predictions come out as ``rmse_baseline`` — effectively equal to
    ``rmse_persistence`` — and ARIMA predictions as ``rmse_arima``).
    ``rmse_persistence`` is kept as a shared reference column.
    """
    out = df.copy()
    rename_map = {"rmse_model": f"rmse_{suffix}"}

    for col in ("skill_rmse", "vr", "kge", "kge_r",
                "kge_alpha", "kge_beta", "skill_vp"):
        if col in out.columns:
            rename_map[col] = f"{col}_{suffix}"

    out = out.rename(columns=rename_map)

    # Also expose unprefixed-friendly aliases requested by the task spec
    # (alpha_<suffix>, beta_<suffix>, r_<suffix>) so downstream code can
    # read either naming convention.
    alias_map = {}
    if f"kge_alpha_{suffix}" in out.columns:
        alias_map[f"alpha_{suffix}"] = out[f"kge_alpha_{suffix}"]
    if f"kge_beta_{suffix}" in out.columns:
        alias_map[f"beta_{suffix}"] = out[f"kge_beta_{suffix}"]
    if f"kge_r_{suffix}" in out.columns:
        alias_map[f"r_{suffix}"] = out[f"kge_r_{suffix}"]
    for k, v in alias_map.items():
        out[k] = v

    return out


def _load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(f"Missing {BASELINE_CSV}")
    if not ARIMA_CSV.exists():
        raise FileNotFoundError(f"Missing {ARIMA_CSV}")

    baseline = pd.read_csv(BASELINE_CSV)
    arima = pd.read_csv(ARIMA_CSV)

    for name, df in (("baseline", baseline), ("arima", arima)):
        for key in KEYS:
            if key not in df.columns:
                raise ValueError(f"{name} CSV missing required key column '{key}'")
        if "rmse_model" not in df.columns:
            raise ValueError(f"{name} CSV missing 'rmse_model' column")
        if "rmse_persistence" not in df.columns:
            raise ValueError(f"{name} CSV missing 'rmse_persistence' column")

    base_keys = baseline[KEYS].sort_values(KEYS).reset_index(drop=True)
    arima_keys = arima[KEYS].sort_values(KEYS).reset_index(drop=True)
    if not base_keys.equals(arima_keys):
        print("Warning: baseline and ARIMA have different (station, horizon) "
              "key sets. Only the intersection will be kept.")

    return baseline, arima


def build_summary() -> pd.DataFrame:
    baseline, arima = _load_and_validate()

    print(f"Loaded baseline rows: {len(baseline)}")
    print(f"Loaded ARIMA rows:    {len(arima)}")

    baseline_r = _rename_with_suffix(baseline, "baseline")
    arima_r = _rename_with_suffix(arima, "arima")

    # rmse_persistence is identical across files by construction; keep
    # it once (taken from the ARIMA frame) to avoid a _x/_y collision.
    baseline_r = baseline_r.drop(columns=["rmse_persistence"])

    merged = baseline_r.merge(arima_r, on=KEYS, how="inner",
                              validate="one_to_one")
    print(f"Merged rows:          {len(merged)}")

    merged["delta_rmse"] = merged["rmse_arima"] - merged["rmse_persistence"]
    merged["better_than_persistence"] = merged["rmse_arima"] < merged["rmse_persistence"]
    merged["variance_collapse_flag"] = merged["alpha_arima"] < VARIANCE_COLLAPSE_THRESHOLD
    merged["severe_variance_collapse_flag"] = merged["alpha_arima"] < SEVERE_VARIANCE_COLLAPSE_THRESHOLD
    merged["overdispersion_flag"] = merged["alpha_arima"] > OVERDISPERSION_THRESHOLD

    ordered_cols = (
        KEYS
        + ["rmse_persistence", "rmse_baseline", "rmse_arima", "delta_rmse",
           "better_than_persistence",
           "skill_rmse_baseline", "skill_rmse_arima",
           "vr_baseline", "vr_arima",
           "kge_baseline", "kge_arima",
           "r_baseline", "r_arima",
           "alpha_baseline", "alpha_arima",
           "beta_baseline", "beta_arima",
           "skill_vp_baseline", "skill_vp_arima",
           "variance_collapse_flag",
           "severe_variance_collapse_flag",
           "overdispersion_flag"]
    )
    ordered_cols = [c for c in ordered_cols if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in ordered_cols]
    merged = merged[ordered_cols + other_cols]

    merged = merged.sort_values(KEYS).reset_index(drop=True)
    return merged


def _fmt_horizons(horizons) -> str:
    horizons = sorted(int(h) for h in horizons)
    return ", ".join(f"h={h}" for h in horizons) if horizons else "none"


def write_findings(summary: pd.DataFrame) -> None:
    lines = []
    lines.append("Paper C - ARIMA vs persistence: key findings")
    lines.append("=" * 48)
    lines.append("")
    lines.append("Thresholds: variance collapse alpha<0.95 (severe <0.85), "
                 "overdispersion alpha>1.05.")
    lines.append("")

    for station, block in summary.groupby("station"):
        lines.append(f"[{station}]")
        better = block.loc[block["better_than_persistence"], "horizon"].tolist()
        worse = block.loc[~block["better_than_persistence"], "horizon"].tolist()
        collapse = block.loc[block["variance_collapse_flag"], "horizon"].tolist()
        severe = block.loc[block["severe_variance_collapse_flag"], "horizon"].tolist()
        overdisp = block.loc[block["overdispersion_flag"], "horizon"].tolist()
        both = block.loc[block["better_than_persistence"]
                         & block["variance_collapse_flag"], "horizon"].tolist()

        lines.append(f"  ARIMA beats persistence on RMSE at: {_fmt_horizons(better)}")
        lines.append(f"  ARIMA does not improve RMSE at:      {_fmt_horizons(worse)}")
        lines.append(f"  Variance collapse (alpha<0.95) at:   {_fmt_horizons(collapse)}")
        if severe:
            lines.append(f"    of which severe (alpha<0.85):      {_fmt_horizons(severe)}")
        if overdisp:
            lines.append(f"  Overdispersion (alpha>1.05) at:      {_fmt_horizons(overdisp)}")
        lines.append(f"  RMSE gain AND variance collapse at:  {_fmt_horizons(both)}")
        lines.append("")

    FINDINGS_TXT.write_text("\n".join(lines))
    print(f"Findings written to {FINDINGS_TXT}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_summary()
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Summary written to {SUMMARY_CSV}")

    write_findings(summary)

    print("\n--- Summary preview ---")
    preview_cols = [
        "station", "horizon", "rmse_persistence", "rmse_arima",
        "delta_rmse", "better_than_persistence",
        "alpha_arima", "variance_collapse_flag",
    ]
    preview_cols = [c for c in preview_cols if c in summary.columns]
    print(summary[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
