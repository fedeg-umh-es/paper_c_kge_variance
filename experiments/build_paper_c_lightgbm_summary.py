"""Build the Paper C LightGBM-vs-persistence summary table.

Lightweight post-processing script: loads the two existing metric CSVs
produced by ``run_paper_c_baseline.py`` and ``run_paper_c_lightgbm.py``,
merges them by ``(station, horizon)``, adds diagnostic derived columns,
and writes a summary CSV plus a short human-readable findings file.

Inputs
------
- results/metrics_baseline_by_horizon.csv
- results/metrics_lightgbm_by_horizon.csv

Outputs
-------
- results/paper_c_lightgbm_vs_persistence_summary.csv
- results/paper_c_lightgbm_key_findings.txt

Only ``pandas`` and ``pathlib`` are used. No models are retrained.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

BASELINE_CSV = RESULTS_DIR / "metrics_baseline_by_horizon.csv"
LIGHTGBM_CSV = RESULTS_DIR / "metrics_lightgbm_by_horizon.csv"

SUMMARY_CSV = RESULTS_DIR / "paper_c_lightgbm_vs_persistence_summary.csv"
FINDINGS_TXT = RESULTS_DIR / "paper_c_lightgbm_key_findings.txt"

KEYS = ["station", "horizon"]

VARIANCE_COLLAPSE_THRESHOLD = 0.95
SEVERE_VARIANCE_COLLAPSE_THRESHOLD = 0.85
OVERDISPERSION_THRESHOLD = 1.05


def _rename_with_suffix(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    out = df.copy()
    rename_map = {"rmse_model": f"rmse_{suffix}"}
    for col in ("skill_rmse", "vr", "kge", "kge_r", "kge_alpha", "kge_beta", "skill_vp"):
        if col in out.columns:
            rename_map[col] = f"{col}_{suffix}"
    out = out.rename(columns=rename_map)
    # Expose short aliases matching the task spec
    for long, short in (("kge_alpha", "alpha"), ("kge_beta", "beta"), ("kge_r", "r")):
        long_suf = f"{long}_{suffix}"
        if long_suf in out.columns:
            out[f"{short}_{suffix}"] = out[long_suf]
    return out


def _load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (BASELINE_CSV, LIGHTGBM_CSV):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

    baseline = pd.read_csv(BASELINE_CSV)
    lgbm = pd.read_csv(LIGHTGBM_CSV)

    for name, df in (("baseline", baseline), ("lightgbm", lgbm)):
        for key in KEYS:
            if key not in df.columns:
                raise ValueError(f"{name} CSV missing key column '{key}'")
        if "rmse_model" not in df.columns:
            raise ValueError(f"{name} CSV missing 'rmse_model'")
        if "rmse_persistence" not in df.columns:
            raise ValueError(f"{name} CSV missing 'rmse_persistence'")

    base_keys = baseline[KEYS].sort_values(KEYS).reset_index(drop=True)
    lgbm_keys = lgbm[KEYS].sort_values(KEYS).reset_index(drop=True)
    if not base_keys.equals(lgbm_keys):
        print("Warning: baseline and LightGBM have different (station, horizon) "
              "key sets. Only the intersection will be kept.")

    return baseline, lgbm


def build_summary() -> pd.DataFrame:
    baseline, lgbm = _load_and_validate()
    print(f"Loaded baseline rows:  {len(baseline)}")
    print(f"Loaded LightGBM rows:  {len(lgbm)}")

    baseline_r = _rename_with_suffix(baseline, "baseline")
    lgbm_r = _rename_with_suffix(lgbm, "lightgbm")

    # rmse_persistence is the same in both; keep it once from LightGBM frame
    baseline_r = baseline_r.drop(columns=["rmse_persistence"])

    merged = baseline_r.merge(lgbm_r, on=KEYS, how="inner", validate="one_to_one")
    print(f"Merged rows:           {len(merged)}")

    merged["delta_rmse"] = merged["rmse_lightgbm"] - merged["rmse_persistence"]
    merged["better_than_persistence"] = merged["rmse_lightgbm"] < merged["rmse_persistence"]
    merged["variance_collapse_flag"] = merged["alpha_lightgbm"] < VARIANCE_COLLAPSE_THRESHOLD
    merged["severe_variance_collapse_flag"] = (
        merged["alpha_lightgbm"] < SEVERE_VARIANCE_COLLAPSE_THRESHOLD
    )
    merged["overdispersion_flag"] = merged["alpha_lightgbm"] > OVERDISPERSION_THRESHOLD

    ordered_cols = (
        KEYS
        + ["rmse_persistence", "rmse_baseline", "rmse_lightgbm", "delta_rmse",
           "better_than_persistence",
           "skill_rmse_baseline", "skill_rmse_lightgbm",
           "vr_baseline", "vr_lightgbm",
           "kge_baseline", "kge_lightgbm",
           "r_baseline", "r_lightgbm",
           "alpha_baseline", "alpha_lightgbm",
           "beta_baseline", "beta_lightgbm",
           "skill_vp_baseline", "skill_vp_lightgbm",
           "variance_collapse_flag",
           "severe_variance_collapse_flag",
           "overdispersion_flag"]
    )
    ordered_cols = [c for c in ordered_cols if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in ordered_cols]
    merged = merged[ordered_cols + other_cols]
    return merged.sort_values(KEYS).reset_index(drop=True)


def _fmt_horizons(horizons) -> str:
    horizons = sorted(int(h) for h in horizons)
    return ", ".join(f"h={h}" for h in horizons) if horizons else "none"


def write_findings(summary: pd.DataFrame) -> None:
    lines = []
    lines.append("Paper C - LightGBM vs persistence: key findings")
    lines.append("=" * 50)
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
        both = block.loc[
            block["better_than_persistence"] & block["variance_collapse_flag"],
            "horizon"
        ].tolist()

        lines.append(f"  LightGBM beats persistence on RMSE at:  {_fmt_horizons(better)}")
        lines.append(f"  LightGBM does not improve RMSE at:       {_fmt_horizons(worse)}")
        lines.append(f"  Variance collapse (alpha<0.95) at:        {_fmt_horizons(collapse)}")
        if severe:
            lines.append(f"    of which severe (alpha<0.85):           {_fmt_horizons(severe)}")
        if overdisp:
            lines.append(f"  Overdispersion (alpha>1.05) at:           {_fmt_horizons(overdisp)}")
        lines.append(f"  RMSE gain AND variance collapse at:       {_fmt_horizons(both)}")
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
        "station", "horizon", "rmse_persistence", "rmse_lightgbm",
        "delta_rmse", "better_than_persistence",
        "alpha_lightgbm", "variance_collapse_flag",
    ]
    preview_cols = [c for c in preview_cols if c in summary.columns]
    print(summary[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
