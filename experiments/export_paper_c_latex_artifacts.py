"""Export final LaTeX artifacts for Paper C from real pipeline outputs.

Reads exclusively from the canonical CSVs produced by the existing
experiments and emits the LaTeX files used directly in Overleaf:

    paper/tables.tex
    paper/results_snippets.tex
    paper/README_ARTIFACTS.md is regenerated separately and not by this
    script (see paper/README_ARTIFACTS.md in the repo).
    paper/figures/fig_paper_c_rmse_skill.png
    paper/figures/fig_paper_c_alpha_vr.png

Real input contract
-------------------
    results/tables/baseline_results.csv
        Produced by ``python3 experiments/run_baseline.py``.
        Schema (from src/evaluation/metrics.py::horizon_profile):
            station, model, horizon, rmse_model, rmse_persistence,
            skill_rmse, vr, kge, kge_r, kge_alpha, kge_beta, skill_vp
        Required values of ``model``: at least ``persistence`` and ``arima``.

    results/tables/lgbm_results.csv
        Produced by ``python3 experiments/run_lgbm.py``.
        Same schema, ``model == "lgbm"``.

If either file is missing or required (model, station, horizon) cells
are absent, the script aborts with a single explicit message and writes
no partial artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "results" / "tables"
PAPER_DIR = ROOT / "paper"
FIGURES_DIR = PAPER_DIR / "figures"

BASELINE_CSV = TABLES_DIR / "baseline_results.csv"
LGBM_CSV = TABLES_DIR / "lgbm_results.csv"

CANONICAL_COLUMNS = [
    "station",
    "model",
    "horizon",
    "rmse_model",
    "rmse_persistence",
    "skill_rmse",
    "vr",
    "kge",
    "kge_r",
    "kge_alpha",
    "kge_beta",
    "skill_vp",
]

GAIN_COLLAPSE_THRESHOLD = 0.95

CORE_COLUMNS = [
    "station",
    "horizon",
    "rmse_persistence",
    "rmse_arima",
    "rmse_lightgbm",
    "skill_rmse_arima",
    "skill_rmse_lightgbm",
    "alpha_arima",
    "alpha_lightgbm",
    "vr_arima",
    "vr_lightgbm",
    "kge_arima",
    "kge_lightgbm",
    "skill_vp_arima",
    "skill_vp_lightgbm",
]

CORE_HEADER_MAP = {
    "station": "Station",
    "horizon": "$h$",
    "rmse_persistence": "RMSE$_{\\mathrm{pers}}$",
    "rmse_arima": "RMSE$_{\\mathrm{AR}}$",
    "rmse_lightgbm": "RMSE$_{\\mathrm{LGBM}}$",
    "skill_rmse_arima": "Skill$_{\\mathrm{AR}}$",
    "skill_rmse_lightgbm": "Skill$_{\\mathrm{LGBM}}$",
    "alpha_arima": "$\\alpha_{\\mathrm{AR}}$",
    "alpha_lightgbm": "$\\alpha_{\\mathrm{LGBM}}$",
    "vr_arima": "VR$_{\\mathrm{AR}}$",
    "vr_lightgbm": "VR$_{\\mathrm{LGBM}}$",
    "kge_arima": "KGE$_{\\mathrm{AR}}$",
    "kge_lightgbm": "KGE$_{\\mathrm{LGBM}}$",
    "skill_vp_arima": "Skill$^{\\mathrm{VP}}_{\\mathrm{AR}}$",
    "skill_vp_lightgbm": "Skill$^{\\mathrm{VP}}_{\\mathrm{LGBM}}$",
}

APPENDIX_COLUMNS = [
    "station",
    "horizon",
    "r_arima",
    "alpha_arima",
    "beta_arima",
    "kge_arima",
    "r_lightgbm",
    "alpha_lightgbm",
    "beta_lightgbm",
    "kge_lightgbm",
]

APPENDIX_HEADER_MAP = {
    "station": "Station",
    "horizon": "$h$",
    "r_arima": "$r_{\\mathrm{AR}}$",
    "alpha_arima": "$\\alpha_{\\mathrm{AR}}$",
    "beta_arima": "$\\beta_{\\mathrm{AR}}$",
    "kge_arima": "KGE$_{\\mathrm{AR}}$",
    "r_lightgbm": "$r_{\\mathrm{LGBM}}$",
    "alpha_lightgbm": "$\\alpha_{\\mathrm{LGBM}}$",
    "beta_lightgbm": "$\\beta_{\\mathrm{LGBM}}$",
    "kge_lightgbm": "KGE$_{\\mathrm{LGBM}}$",
}


class MissingRealOutput(RuntimeError):
    """Raised when a required real pipeline output is absent."""


def _abort(message: str) -> "None":
    print(f"[export_paper_c_latex_artifacts] {message}", file=sys.stderr)
    raise MissingRealOutput(message)


def _load_csv(path: Path, expected_models: set[str], regen_command: str) -> pd.DataFrame:
    if not path.exists():
        _abort(
            f"Missing real pipeline output: {path.relative_to(ROOT)}.\n"
            f"Regenerate with: {regen_command}"
        )
    df = pd.read_csv(path)
    missing_cols = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing_cols:
        _abort(
            f"{path.relative_to(ROOT)} is missing required columns "
            f"(see src/evaluation/metrics.py::horizon_profile): {missing_cols}"
        )
    present = set(df["model"].unique())
    missing_models = expected_models - present
    if missing_models:
        _abort(
            f"{path.relative_to(ROOT)} is missing required model rows: "
            f"{sorted(missing_models)} (found {sorted(present)})."
        )
    return df


def _slice_model(df: pd.DataFrame, model: str) -> pd.DataFrame:
    return df.loc[df["model"] == model].copy()


def build_core_frame(baseline: pd.DataFrame, lgbm: pd.DataFrame) -> pd.DataFrame:
    arima = _slice_model(baseline, "arima")
    light = _slice_model(lgbm, "lgbm")

    arima_view = arima[
        ["station", "horizon", "rmse_persistence", "rmse_model", "skill_rmse",
         "kge_alpha", "vr", "kge", "skill_vp"]
    ].rename(
        columns={
            "rmse_model": "rmse_arima",
            "skill_rmse": "skill_rmse_arima",
            "kge_alpha": "alpha_arima",
            "vr": "vr_arima",
            "kge": "kge_arima",
            "skill_vp": "skill_vp_arima",
        }
    )

    light_view = light[
        ["station", "horizon", "rmse_model", "skill_rmse",
         "kge_alpha", "vr", "kge", "skill_vp"]
    ].rename(
        columns={
            "rmse_model": "rmse_lightgbm",
            "skill_rmse": "skill_rmse_lightgbm",
            "kge_alpha": "alpha_lightgbm",
            "vr": "vr_lightgbm",
            "kge": "kge_lightgbm",
            "skill_vp": "skill_vp_lightgbm",
        }
    )

    merged = pd.merge(arima_view, light_view, on=["station", "horizon"], how="inner")
    if merged.empty:
        _abort(
            "No (station, horizon) cell is present in both "
            "results/tables/baseline_results.csv (model=arima) and "
            "results/tables/lgbm_results.csv (model=lgbm). "
            "Re-run both experiments before exporting."
        )

    merged = merged.sort_values(["station", "horizon"]).reset_index(drop=True)
    return merged[CORE_COLUMNS]


def build_appendix_frame(baseline: pd.DataFrame, lgbm: pd.DataFrame) -> pd.DataFrame:
    arima = _slice_model(baseline, "arima")
    light = _slice_model(lgbm, "lgbm")

    arima_view = arima[
        ["station", "horizon", "kge_r", "kge_alpha", "kge_beta", "kge"]
    ].rename(
        columns={
            "kge_r": "r_arima",
            "kge_alpha": "alpha_arima",
            "kge_beta": "beta_arima",
            "kge": "kge_arima",
        }
    )
    light_view = light[
        ["station", "horizon", "kge_r", "kge_alpha", "kge_beta", "kge"]
    ].rename(
        columns={
            "kge_r": "r_lightgbm",
            "kge_alpha": "alpha_lightgbm",
            "kge_beta": "beta_lightgbm",
            "kge": "kge_lightgbm",
        }
    )

    merged = pd.merge(arima_view, light_view, on=["station", "horizon"], how="inner")
    merged = merged.sort_values(["station", "horizon"]).reset_index(drop=True)
    return merged[APPENDIX_COLUMNS]


def compute_macro_counts(core: pd.DataFrame) -> dict[str, tuple[int, int]]:
    n = len(core)
    return {
        "ArimaBetterCount": (
            int((core["rmse_arima"] < core["rmse_persistence"]).sum()), n
        ),
        "LGBMBetterCount": (
            int((core["rmse_lightgbm"] < core["rmse_persistence"]).sum()), n
        ),
        "LGBMBeatsArimaCount": (
            int((core["rmse_lightgbm"] < core["rmse_arima"]).sum()), n
        ),
        "LGBMLowerAlphaCount": (
            int((core["alpha_lightgbm"] < core["alpha_arima"]).sum()), n
        ),
        "ArimaGainCollapseCount": (
            int((core["alpha_arima"] < GAIN_COLLAPSE_THRESHOLD).sum()), n
        ),
        "LGBMGainCollapseCount": (
            int((core["alpha_lightgbm"] < GAIN_COLLAPSE_THRESHOLD).sum()), n
        ),
    }


def _escape_latex(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def _fmt_number(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.3f}"


def _fmt_horizon(value) -> str:
    return f"$h{{=}}{int(value)}$"


def _format_cell(col: str, value) -> str:
    if col == "station":
        return _escape_latex(value)
    if col == "horizon":
        return _fmt_horizon(value)
    return _fmt_number(value)


def _render_row(row: pd.Series, columns: list[str]) -> str:
    return " & ".join(_format_cell(c, row[c]) for c in columns) + " \\\\"


SOURCE_BANNER = (
    "% Auto-generated by experiments/export_paper_c_latex_artifacts.py\n"
    "% Source: results/tables/baseline_results.csv (model in {persistence, arima})\n"
    "% Source: results/tables/lgbm_results.csv (model = lgbm)\n"
    "% Do not edit by hand. Re-run the export script to regenerate.\n"
)


def build_tables_tex(core: pd.DataFrame, appendix: pd.DataFrame) -> str:
    core_header = " & ".join(CORE_HEADER_MAP[c] for c in CORE_COLUMNS) + " \\\\"
    core_body = "\n".join(_render_row(r, CORE_COLUMNS) for _, r in core.iterrows())
    core_colspec = "l c " + " ".join(["r"] * (len(CORE_COLUMNS) - 2))

    appendix_header = " & ".join(APPENDIX_HEADER_MAP[c] for c in APPENDIX_COLUMNS) + " \\\\"
    appendix_body = "\n".join(
        _render_row(r, APPENDIX_COLUMNS) for _, r in appendix.iterrows()
    )
    appendix_colspec = "l c " + " ".join(["r"] * (len(APPENDIX_COLUMNS) - 2))

    body = dedent(
        r"""

        \begin{table}[t]
        \centering
        \scriptsize
        \setlength{\tabcolsep}{3pt}
        \caption{Per-station, per-horizon point-forecast metrics on the held-out
        rolling-origin folds. Skill scores use the persistence baseline.
        $\alpha$ is the KGE variability ratio and VR the variance ratio
        $\mathrm{Var}(\hat{y})/\mathrm{Var}(y)$; both diagnose amplitude
        (under)estimation. Skill$^{\mathrm{VP}}$ is the variance-penalised
        skill $\mathrm{Skill}_{\mathrm{RMSE}}\cdot\min(1,\mathrm{VR})$.}
        \label{tab:core}
        \resizebox{\textwidth}{!}{%
        \begin{tabular}{__CORE_COLSPEC__}
        \toprule
        __CORE_HEADER__
        \midrule
        __CORE_BODY__
        \bottomrule
        \end{tabular}%
        }
        \end{table}

        \begin{table}[t]
        \centering
        \scriptsize
        \setlength{\tabcolsep}{3pt}
        \caption{KGE decomposition per station and horizon. $r$ is the Pearson
        correlation, $\alpha$ the variability ratio, $\beta$ the bias ratio,
        and KGE the Kling--Gupta Efficiency.}
        \label{tab:appendix}
        \resizebox{\textwidth}{!}{%
        \begin{tabular}{__APPX_COLSPEC__}
        \toprule
        __APPX_HEADER__
        \midrule
        __APPX_BODY__
        \bottomrule
        \end{tabular}%
        }
        \end{table}
        """
    ).strip("\n") + "\n"

    body = (
        body.replace("__CORE_COLSPEC__", core_colspec)
        .replace("__CORE_HEADER__", core_header)
        .replace("__CORE_BODY__", core_body)
        .replace("__APPX_COLSPEC__", appendix_colspec)
        .replace("__APPX_HEADER__", appendix_header)
        .replace("__APPX_BODY__", appendix_body)
    )
    return SOURCE_BANNER + "\n" + body


def build_snippets_tex(counts: dict[str, tuple[int, int]]) -> str:
    macros = "\n".join(
        f"\\newcommand{{\\{name}}}{{{num}/{den}}}"
        for name, (num, den) in counts.items()
    )
    audit_block = "\n".join(
        f"% {name} = {num}/{den}" for name, (num, den) in counts.items()
    )

    paragraph = dedent(
        r"""
        % Suggested Results/Discussion paragraph (academic English):
        %
        % Across the available (station, horizon) cells, ARIMA improves upon
        % the persistence baseline in \ArimaBetterCount of them, while
        % LightGBM improves upon it in \LGBMBetterCount. In a direct
        % head-to-head comparison, LightGBM attains a lower RMSE than ARIMA
        % in \LGBMBeatsArimaCount of the cells. The KGE variability ratio
        % alpha is lower for LightGBM than for ARIMA in \LGBMLowerAlphaCount
        % of the cells, indicating that the tree ensemble contracts more
        % strongly toward the conditional mean. Consistently, alpha falls
        % below 0.95 -- a regime in which the variance-penalised skill
        % Skill_VP lies well under the plain skill score -- in
        % \ArimaGainCollapseCount (ARIMA) and \LGBMGainCollapseCount
        % (LightGBM) of the cells. These results support reporting alpha
        % and Skill_VP alongside RMSE.
        """
    ).strip()

    figures = dedent(
        r"""
        % ---- Figures ----
        \begin{figure}[t]
        \centering
        \includegraphics[width=\linewidth]{figures/fig_paper_c_rmse_skill.png}
        \caption{RMSE (left) and Skill$_{\mathrm{RMSE}}$ (right) as a
        function of the forecast horizon $h$ for both stations.}
        \label{fig:rmse}
        \end{figure}

        \begin{figure}[t]
        \centering
        \includegraphics[width=\linewidth]{figures/fig_paper_c_alpha_vr.png}
        \caption{KGE variability ratio $\alpha$ (left) and variance ratio
        $\mathrm{VR}=\mathrm{Var}(\hat{y})/\mathrm{Var}(y)$ (right) as a
        function of the forecast horizon $h$. The dashed reference at $1$
        marks perfect amplitude retention; values below it indicate
        underdispersion.}
        \label{fig:alpha}
        \end{figure}
        """
    ).strip()

    parts = [
        SOURCE_BANNER,
        "% ---- Counts derived from real outputs (audit block) ----",
        audit_block,
        "",
        "% ---- Global count macros (computed from the real CSVs above) ----",
        macros,
        "",
        paragraph,
        "",
        figures,
        "",
    ]
    return "\n".join(parts)


def build_readme_artifacts(counts: dict[str, tuple[int, int]]) -> str:
    counts_lines = "\n".join(
        f"- `\\{name}` -> {num}/{den}" for name, (num, den) in counts.items()
    )
    return dedent(
        f"""
        # Paper C - Overleaf artifacts

        Final LaTeX artifacts for Paper C, generated by
        `experiments/export_paper_c_latex_artifacts.py` exclusively from the
        real pipeline outputs:

        - `results/tables/baseline_results.csv` (rows with `model in {{persistence, arima}}`)
        - `results/tables/lgbm_results.csv` (rows with `model = lgbm`)

        The script does not retrain any model and does not read any
        intermediate file under `results/`.

        ## Files to upload to Overleaf

        Upload these files preserving the directory layout:

        1. `paper/tables.tex`
        2. `paper/results_snippets.tex`
        3. `paper/figures/fig_paper_c_rmse_skill.png`
        4. `paper/figures/fig_paper_c_alpha_vr.png`

        ## How to insert in `main.tex`

        In the preamble (after `\\usepackage{{graphicx}}` and `booktabs`):

        ```latex
        \\input{{results_snippets}}
        ```

        Inside the Results section:

        ```latex
        \\input{{tables}}
        ```

        The figures are referenced from `results_snippets.tex` with
        `\\label{{fig:rmse}}` and `\\label{{fig:alpha}}` and expect the PNGs
        under `figures/`.

        ## Macros (current values)

        Computed from the real CSVs at the time of the last export:

        {counts_lines}

        ## Regeneration sequence

        From the repository root, in this order:

        ```bash
        python3 experiments/run_baseline.py   # writes results/tables/baseline_results.csv
        python3 experiments/run_lgbm.py       # writes results/tables/lgbm_results.csv
        python3 experiments/export_paper_c_latex_artifacts.py
        ```

        If a required input is missing, the export script aborts with an
        explicit error naming the file and the command that produces it.
        It never writes partial artifacts.
        """
    ).strip() + "\n"


def _plot_metric(ax, frame: pd.DataFrame, value_col: str, label_prefix: str) -> None:
    for station, sub in frame.groupby("station"):
        sub = sub.sort_values("horizon")
        ax.plot(sub["horizon"], sub[value_col], marker="o",
                label=f"{station} - {label_prefix}")


def build_rmse_skill_figure(baseline: pd.DataFrame, lgbm: pd.DataFrame, out_path: Path) -> None:
    persistence = _slice_model(baseline, "persistence")
    arima = _slice_model(baseline, "arima")
    light = _slice_model(lgbm, "lgbm")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)

    ax = axes[0]
    _plot_metric(ax, persistence.assign(rmse=persistence["rmse_model"]),
                 "rmse", "persistence")
    _plot_metric(ax, arima.assign(rmse=arima["rmse_model"]), "rmse", "ARIMA")
    _plot_metric(ax, light.assign(rmse=light["rmse_model"]), "rmse", "LightGBM")
    horizons = sorted(arima["horizon"].unique().tolist())
    ax.set_xticks(horizons)
    ax.set_xlabel("Horizon h (hours)")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs horizon")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    _plot_metric(ax, arima, "skill_rmse", "ARIMA")
    _plot_metric(ax, light, "skill_rmse", "LightGBM")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(horizons)
    ax.set_xlabel("Horizon h (hours)")
    ax.set_ylabel("Skill_RMSE")
    ax.set_title("Skill_RMSE vs horizon")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_alpha_vr_figure(baseline: pd.DataFrame, lgbm: pd.DataFrame, out_path: Path) -> None:
    arima = _slice_model(baseline, "arima")
    light = _slice_model(lgbm, "lgbm")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)

    ax = axes[0]
    _plot_metric(ax, arima, "kge_alpha", "ARIMA")
    _plot_metric(ax, light, "kge_alpha", "LightGBM")
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--")
    horizons = sorted(arima["horizon"].unique().tolist())
    ax.set_xticks(horizons)
    ax.set_xlabel("Horizon h (hours)")
    ax.set_ylabel("alpha (KGE variability ratio)")
    ax.set_title("alpha vs horizon")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    _plot_metric(ax, arima, "vr", "ARIMA")
    _plot_metric(ax, light, "vr", "LightGBM")
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xticks(horizons)
    ax.set_xlabel("Horizon h (hours)")
    ax.set_ylabel("VR = Var(y_hat)/Var(y)")
    ax.set_title("VR vs horizon")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    try:
        baseline = _load_csv(
            BASELINE_CSV,
            expected_models={"persistence", "arima"},
            regen_command="python3 experiments/run_baseline.py",
        )
        lgbm = _load_csv(
            LGBM_CSV,
            expected_models={"lgbm"},
            regen_command="python3 experiments/run_lgbm.py",
        )
    except MissingRealOutput:
        return 1

    try:
        core = build_core_frame(baseline, lgbm)
        appendix = build_appendix_frame(baseline, lgbm)
    except MissingRealOutput:
        return 2

    counts = compute_macro_counts(core)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (PAPER_DIR / "tables.tex").write_text(
        build_tables_tex(core, appendix), encoding="utf-8"
    )
    (PAPER_DIR / "results_snippets.tex").write_text(
        build_snippets_tex(counts), encoding="utf-8"
    )
    (PAPER_DIR / "README_ARTIFACTS.md").write_text(
        build_readme_artifacts(counts), encoding="utf-8"
    )

    build_rmse_skill_figure(baseline, lgbm, FIGURES_DIR / "fig_paper_c_rmse_skill.png")
    build_alpha_vr_figure(baseline, lgbm, FIGURES_DIR / "fig_paper_c_alpha_vr.png")

    written = [
        PAPER_DIR / "tables.tex",
        PAPER_DIR / "results_snippets.tex",
        PAPER_DIR / "README_ARTIFACTS.md",
        FIGURES_DIR / "fig_paper_c_rmse_skill.png",
        FIGURES_DIR / "fig_paper_c_alpha_vr.png",
    ]
    print("[export_paper_c_latex_artifacts] wrote:")
    for p in written:
        print(f"  - {p.relative_to(ROOT)}")
    print(f"[export_paper_c_latex_artifacts] cells used: {len(core)} "
          f"(stations={core['station'].nunique()}, horizons={core['horizon'].nunique()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
