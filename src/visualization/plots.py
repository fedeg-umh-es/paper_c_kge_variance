"""Publication-quality figures for the MATCOM paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend for CI/scripting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

FIGURES_DIR = Path(__file__).parent.parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "persistence": "#6c757d",
    "arima":       "#0077b6",
    "lstm":        "#e07a5f",
    "lgbm":        "#3d405b",
}

MARKER = {
    "persistence": "s",
    "arima":       "o",
    "lstm":        "^",
    "lgbm":        "D",
}


def _save(fig: plt.Figure, name: str, dpi: int = 300) -> Path:
    out = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    out_png = FIGURES_DIR / f"{name}.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out


def plot_horizon_metric(
    results_dict: dict[str, pd.DataFrame],
    metric: str = "skill_rmse",
    ylabel: str | None = None,
    title: str | None = None,
    filename: str = "horizon_metric",
) -> Path:
    """Line plot of a single metric across horizons for all models.

    Parameters
    ----------
    results_dict:
        {model_name: horizon_profile_df}.
    metric:
        Column to plot.
    ylabel, title:
        Axis labels; defaults inferred from metric name.
    filename:
        Output filename (without extension).
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for model_name, df in results_dict.items():
        horizons = df["horizon"].values
        vals = df[metric].values
        color = PALETTE.get(model_name, None)
        marker = MARKER.get(model_name, "o")
        ax.plot(horizons, vals, label=model_name, color=color,
                marker=marker, linewidth=1.8, markersize=6)

    ax.set_xlabel("Forecast horizon (h)", fontsize=11)
    ax.set_ylabel(ylabel or metric.replace("_", " ").upper(), fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return _save(fig, filename)


def plot_kge_components(
    results_dict: dict[str, pd.DataFrame],
    filename: str = "kge_components",
) -> Path:
    """Three-panel figure: KGE-r, KGE-alpha, KGE-beta by horizon."""
    components = [("kge_r", "r (correlation)"),
                  ("kge_alpha", r"$\alpha$ (variability)"),
                  ("kge_beta",  r"$\beta$ (bias)")]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    for ax, (col, label) in zip(axes, components):
        for model_name, df in results_dict.items():
            horizons = df["horizon"].values
            vals = df[col].values
            ax.plot(horizons, vals, label=model_name,
                    color=PALETTE.get(model_name),
                    marker=MARKER.get(model_name, "o"),
                    linewidth=1.8, markersize=5)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Horizon (h)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("KGE Component Profiles by Forecast Horizon", fontsize=12)
    fig.tight_layout()
    return _save(fig, filename)


def plot_variance_retention(
    results_dict: dict[str, pd.DataFrame],
    filename: str = "variance_retention",
) -> Path:
    """VR profile plot with collapse threshold line at VR=1."""
    return plot_horizon_metric(
        results_dict,
        metric="vr",
        ylabel="Variance Retention (VR)",
        title="Variance Retention by Forecast Horizon",
        filename=filename,
    )


def plot_skill_vp(
    results_dict: dict[str, pd.DataFrame],
    filename: str = "skill_vp",
) -> Path:
    """Skill_VP vs plain Skill_RMSE side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for metric, ax, title_str in [
        ("skill_rmse", axes[0], "Skill$_{RMSE}$"),
        ("skill_vp",   axes[1], "Skill$_{VP}$ (variance-penalised)"),
    ]:
        for model_name, df in results_dict.items():
            horizons = df["horizon"].values
            vals = df[metric].values
            ax.plot(horizons, vals, label=model_name,
                    color=PALETTE.get(model_name),
                    marker=MARKER.get(model_name, "o"),
                    linewidth=1.8, markersize=5)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_xlabel("Horizon (h)", fontsize=10)
        ax.set_title(title_str, fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    axes[0].set_ylabel("Skill", fontsize=10)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return _save(fig, filename)
