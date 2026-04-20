"""Build the Paper C master comparison package.

Loads the three existing metric CSVs (persistence baseline, ARIMA,
LightGBM), merges them by (station, horizon), adds derived diagnostic
flags, and writes:

  results/paper_c_master_comparison.csv  — full wide table with all flags
  results/paper_c_table_core.csv         — compact table for the paper body
  results/paper_c_table_appendix.csv     — extended table for the appendix
  results/paper_c_main_findings.txt      — data-only narrative summary
  results/fig_paper_c_rmse_skill.png     — RMSE by horizon per model / station
  results/fig_paper_c_alpha_vr.png       — alpha by horizon per model / station

Only pandas, pathlib, and matplotlib are used. No models are retrained.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

BASELINE_CSV = RESULTS / "metrics_baseline_by_horizon.csv"
ARIMA_CSV    = RESULTS / "metrics_arima_by_horizon.csv"
LGBM_CSV     = RESULTS / "metrics_lightgbm_by_horizon.csv"

MASTER_CSV   = RESULTS / "paper_c_master_comparison.csv"
CORE_CSV     = RESULTS / "paper_c_table_core.csv"
APPENDIX_CSV = RESULTS / "paper_c_table_appendix.csv"
FINDINGS_TXT = RESULTS / "paper_c_main_findings.txt"
FIG_RMSE     = RESULTS / "fig_paper_c_rmse_skill.png"
FIG_ALPHA    = RESULTS / "fig_paper_c_alpha_vr.png"

KEYS = ["station", "horizon"]

VC_THRESHOLD     = 0.95
VC_SEVERE        = 0.85
OD_THRESHOLD     = 1.05


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rename(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Rename all metric columns with *suffix*, expose short alpha/beta/r aliases."""
    out = df.copy()
    rmap = {"rmse_model": f"rmse_{suffix}"}
    for col in ("skill_rmse", "vr", "kge", "kge_r", "kge_alpha", "kge_beta", "skill_vp"):
        if col in out.columns:
            rmap[col] = f"{col}_{suffix}"
    out = out.rename(columns=rmap)
    for long, short in (("kge_alpha", "alpha"), ("kge_beta", "beta"), ("kge_r", "r")):
        src = f"{long}_{suffix}"
        if src in out.columns:
            out[f"{short}_{suffix}"] = out[src]
    return out


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in (BASELINE_CSV, ARIMA_CSV, LGBM_CSV):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")
    base = pd.read_csv(BASELINE_CSV)
    arima = pd.read_csv(ARIMA_CSV)
    lgbm  = pd.read_csv(LGBM_CSV)
    for name, df in (("baseline", base), ("arima", arima), ("lightgbm", lgbm)):
        for k in KEYS + ["rmse_model", "rmse_persistence"]:
            if k not in df.columns:
                raise ValueError(f"{name} CSV missing column '{k}'")
    print(f"Loaded  baseline: {len(base)} rows")
    print(f"Loaded  ARIMA:    {len(arima)} rows")
    print(f"Loaded  LightGBM: {len(lgbm)} rows")
    return base, arima, lgbm


# ---------------------------------------------------------------------------
# build master table
# ---------------------------------------------------------------------------

def build_master() -> pd.DataFrame:
    base, arima, lgbm = _load()

    base_r  = _rename(base,  "baseline")
    arima_r = _rename(arima, "arima")
    lgbm_r  = _rename(lgbm,  "lightgbm")

    # rmse_persistence is identical across all three frames; keep it once
    base_r  = base_r.drop(columns=["rmse_persistence"])
    arima_r = arima_r.drop(columns=["rmse_persistence"])

    m = base_r.merge(arima_r, on=KEYS, how="inner", validate="one_to_one")
    m = m.merge(lgbm_r,  on=KEYS, how="inner", validate="one_to_one")
    print(f"Merged rows:      {len(m)}")

    # derived flags
    m["delta_rmse_arima"]    = m["rmse_arima"]    - m["rmse_persistence"]
    m["delta_rmse_lightgbm"] = m["rmse_lightgbm"] - m["rmse_persistence"]

    m["arima_better_than_persistence"]    = m["rmse_arima"]    < m["rmse_persistence"]
    m["lightgbm_better_than_persistence"] = m["rmse_lightgbm"] < m["rmse_persistence"]

    m["arima_variance_collapse"]    = m["alpha_arima"]    < VC_THRESHOLD
    m["lightgbm_variance_collapse"] = m["alpha_lightgbm"] < VC_THRESHOLD
    m["arima_severe_collapse"]      = m["alpha_arima"]    < VC_SEVERE
    m["lightgbm_severe_collapse"]   = m["alpha_lightgbm"] < VC_SEVERE
    m["arima_overdispersion"]       = m["alpha_arima"]    > OD_THRESHOLD
    m["lightgbm_overdispersion"]    = m["alpha_lightgbm"] > OD_THRESHOLD

    m["arima_rmse_gain_with_collapse"]    = (
        m["arima_better_than_persistence"] & m["arima_variance_collapse"]
    )
    m["lightgbm_rmse_gain_with_collapse"] = (
        m["lightgbm_better_than_persistence"] & m["lightgbm_variance_collapse"]
    )

    return m.sort_values(KEYS).reset_index(drop=True)


# ---------------------------------------------------------------------------
# table slices
# ---------------------------------------------------------------------------

CORE_COLS = [
    "station", "horizon",
    "rmse_persistence", "rmse_arima", "rmse_lightgbm",
    "skill_rmse_arima", "skill_rmse_lightgbm",
    "alpha_arima", "alpha_lightgbm",
    "vr_arima", "vr_lightgbm",
    "kge_arima", "kge_lightgbm",
    "skill_vp_arima", "skill_vp_lightgbm",
]


def write_tables(master: pd.DataFrame) -> None:
    core_cols = [c for c in CORE_COLS if c in master.columns]
    master[core_cols].to_csv(CORE_CSV, index=False)
    print(f"Core table  -> {CORE_CSV}")

    master.to_csv(APPENDIX_CSV, index=False)
    print(f"Appendix    -> {APPENDIX_CSV}")

    master.to_csv(MASTER_CSV, index=False)
    print(f"Master CSV  -> {MASTER_CSV}")


# ---------------------------------------------------------------------------
# findings narrative
# ---------------------------------------------------------------------------

def _fmt_h(horizons) -> str:
    horizons = sorted(int(h) for h in horizons)
    return ", ".join(f"h={h}" for h in horizons) if horizons else "none"


def write_findings(m: pd.DataFrame) -> None:
    n = len(m)
    ar_better   = int(m["arima_better_than_persistence"].sum())
    lgbm_better = int(m["lightgbm_better_than_persistence"].sum())
    ar_gain_vc   = int(m["arima_rmse_gain_with_collapse"].sum())
    lgbm_gain_vc = int(m["lightgbm_rmse_gain_with_collapse"].sum())
    lgbm_beats_arima = int((m["rmse_lightgbm"] < m["rmse_arima"]).sum())
    lgbm_lower_alpha = int((m["alpha_lightgbm"] < m["alpha_arima"]).sum())

    lines = []
    lines += [
        "Paper C — Master comparison: Persistence / ARIMA(2,1,2) / LightGBM",
        "=" * 68,
        "",
        "Thresholds: variance collapse alpha<0.95 (severe <0.85), "
        "overdispersion alpha>1.05.",
        f"Total station×horizon combinations: {n}",
        "",
        "A. Global findings",
        "-" * 36,
        f"  ARIMA beats persistence on RMSE:             {ar_better}/{n} combinations",
        f"  LightGBM beats persistence on RMSE:          {lgbm_better}/{n} combinations",
        f"  ARIMA: RMSE gain AND variance collapse:       {ar_gain_vc}/{n} combinations",
        f"  LightGBM: RMSE gain AND variance collapse:    {lgbm_gain_vc}/{n} combinations",
        f"  LightGBM beats ARIMA on RMSE:                {lgbm_beats_arima}/{n} combinations",
        f"  LightGBM has lower alpha than ARIMA:         {lgbm_lower_alpha}/{n} combinations",
        "",
    ]

    lines += ["B. Per-station breakdown", "-" * 36]
    for station, blk in m.groupby("station"):
        ar_b   = blk.loc[blk["arima_better_than_persistence"],    "horizon"].tolist()
        lgbm_b = blk.loc[blk["lightgbm_better_than_persistence"], "horizon"].tolist()
        ar_vc   = blk.loc[blk["arima_variance_collapse"],    "horizon"].tolist()
        lgbm_vc = blk.loc[blk["lightgbm_variance_collapse"], "horizon"].tolist()
        ar_sv   = blk.loc[blk["arima_severe_collapse"],    "horizon"].tolist()
        lgbm_sv = blk.loc[blk["lightgbm_severe_collapse"], "horizon"].tolist()
        ar_both   = blk.loc[blk["arima_rmse_gain_with_collapse"],    "horizon"].tolist()
        lgbm_both = blk.loc[blk["lightgbm_rmse_gain_with_collapse"], "horizon"].tolist()

        lines += [
            f"  [{station}]",
            f"    ARIMA beats persistence on RMSE:    {_fmt_h(ar_b)}",
            f"    LightGBM beats persistence on RMSE: {_fmt_h(lgbm_b)}",
            f"    ARIMA variance collapse (a<0.95):   {_fmt_h(ar_vc)}",
            f"    LightGBM variance collapse (a<0.95):{_fmt_h(lgbm_vc)}",
        ]
        if ar_sv:
            lines.append(f"      ARIMA severe collapse (a<0.85):   {_fmt_h(ar_sv)}")
        if lgbm_sv:
            lines.append(f"      LightGBM severe collapse (a<0.85):{_fmt_h(lgbm_sv)}")
        lines += [
            f"    ARIMA gain+collapse:                {_fmt_h(ar_both)}",
            f"    LightGBM gain+collapse:             {_fmt_h(lgbm_both)}",
            "",
        ]

    lines += ["C. ARIMA vs LightGBM", "-" * 36]
    lgbm_lower_rmse_rows = m[m["rmse_lightgbm"] < m["rmse_arima"]]
    lgbm_lower_alpha_rows = m[m["alpha_lightgbm"] < m["alpha_arima"]]
    lines += [
        f"  LightGBM achieves lower RMSE than ARIMA in {lgbm_beats_arima}/{n} combinations.",
        f"  LightGBM has lower alpha than ARIMA in {lgbm_lower_alpha}/{n} combinations,",
        f"  suggesting stronger variance shrinkage on average.",
    ]
    if lgbm_lower_alpha >= n // 2:
        lines.append(
            "  Pattern: LightGBM appears more aggressive in variance shrinkage than ARIMA."
        )
    else:
        lines.append(
            "  Pattern: variance shrinkage is comparable between ARIMA and LightGBM."
        )
    lines.append("")

    FINDINGS_TXT.write_text("\n".join(lines))
    print(f"Findings    -> {FINDINGS_TXT}")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(m: pd.DataFrame) -> None:
    stations = sorted(m["station"].unique())
    horizons = sorted(m["horizon"].unique())
    n_st = len(stations)

    # --- fig 1: RMSE by horizon, one panel per station ---
    fig, axes = plt.subplots(1, n_st, figsize=(5 * n_st, 4), sharey=False)
    if n_st == 1:
        axes = [axes]

    for ax, station in zip(axes, stations):
        blk = m[m["station"] == station].sort_values("horizon")
        ax.plot(blk["horizon"], blk["rmse_persistence"], marker="o", label="Persistence")
        ax.plot(blk["horizon"], blk["rmse_arima"],       marker="s", label="ARIMA(2,1,2)")
        ax.plot(blk["horizon"], blk["rmse_lightgbm"],    marker="^", label="LightGBM")
        ax.set_title(station)
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel("RMSE")
        ax.set_xticks(horizons)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("RMSE by horizon — Persistence / ARIMA / LightGBM", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_RMSE, dpi=150)
    plt.close(fig)
    print(f"Figure      -> {FIG_RMSE}")

    # --- fig 2: alpha by horizon, one panel per station ---
    fig, axes = plt.subplots(1, n_st, figsize=(5 * n_st, 4), sharey=False)
    if n_st == 1:
        axes = [axes]

    for ax, station in zip(axes, stations):
        blk = m[m["station"] == station].sort_values("horizon")
        ax.plot(blk["horizon"], blk["alpha_arima"],    marker="s", label="ARIMA(2,1,2)")
        ax.plot(blk["horizon"], blk["alpha_lightgbm"], marker="^", label="LightGBM")
        ax.axhline(VC_THRESHOLD, color="gray",   linestyle="--", linewidth=0.8,
                   label="collapse (0.95)")
        ax.axhline(VC_SEVERE,    color="gray",   linestyle=":",  linewidth=0.8,
                   label="severe (0.85)")
        ax.axhline(OD_THRESHOLD, color="silver", linestyle="--", linewidth=0.8,
                   label="overdisp. (1.05)")
        ax.set_title(station)
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel("alpha (std ratio)")
        ax.set_xticks(horizons)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("KGE alpha (variability ratio) by horizon — ARIMA vs LightGBM", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_ALPHA, dpi=150)
    plt.close(fig)
    print(f"Figure      -> {FIG_ALPHA}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    master = build_master()
    write_tables(master)
    write_findings(master)
    make_figures(master)

    print("\n--- Core table preview ---")
    core_cols = [c for c in CORE_COLS if c in master.columns]
    print(master[core_cols].to_string(index=False))


if __name__ == "__main__":
    main()
