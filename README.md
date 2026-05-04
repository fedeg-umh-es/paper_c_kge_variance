# Paper C: KGE-Based Diagnostic Framework for Variance Collapse Detection in Multi-Horizon Environmental Forecasting

## Scientific Motivation

In environmental forecasting, specifically for pollutants like PM10, models often suffer from variance collapse—a phenomenon where forecasts predict the mean to minimize mean squared error but substantially underestimate natural variability and extreme events. This repository provides a rigorous diagnostic framework to detect such underdispersion. We evaluate the performance of forecasting models across multiple horizons relative to a naive persistence baseline. To uncover structural biases and variability mismatches, we transfer the Kling-Gupta Efficiency (KGE) metric—traditionally used in hydrological modeling—into the air quality domain. In our context, KGE acts strictly as a diagnostic perspective to decompose predictive errors, not as a novel performance metric in itself.

## Repository Structure

```text
paper_c_kge_variance/
├── data/
│   ├── raw/             # Contains original PM10 datasets
│   └── processed/       # Contains unified modeling datasets
├── experiments/         # Scripts for launching baseline and deep learning models
├── paper/               # LaTeX source and documentation materials
├── results/             # Output directory for predictions and metric tables
└── src/
    ├── evaluation/      # Metrics (RMSE, KGE components)
    ├── models/          # Model implementations (Persistence, ARIMA, etc.)
    ├── preprocessing/   # Data unification and scaling logic
    └── visualization/   # Diagnostic plotting 
```

## Datasets

The repository utilizes ambient PM10 measurements from urban stations in Spain:
- **Raw Input 1:** `data/raw/PM10_Madrid_60_hourly.csv`
- **Raw Input 2:** `data/raw/PM10_Valencia_Politecnico_hourly.csv`

Running the preprocessing pipeline standardizes and combines these sources into a unified dataset:
- **Processed Output:** `data/processed/pm10_two_station_hourly.csv`

## Methodology

To ensure strict, leakage-free evaluation, all experiments adhere to the following protocol:
- **Rolling-Origin Evaluation:** We employ an expanding-window (rolling-origin) cross-validation framework to test out-of-sample skill accurately across progressive time steps.
- **Train-Only Preprocessing:** Scalers and imputers are strictly fitted on the training split of every fold to natively prevent data leakage.
- **Forecast Horizons:** Models iteratively generate direct or recursive point forecasts for horizons $h \in \{1, 6, 12, 24\}$ hours.
- **Persistence Baseline:** A naive persistence model serves as the absolute baseline for all computed skill scores, ensuring that learning models genuinely outperform trivial predictions.

## Metrics

We record a comprehensive suite of metrics for component-level evaluation:
- **RMSE:** Root Mean Squared Error.
- **Skill_RMSE:** Relative skill score utilizing the persistence baseline. Computes the percentage of improvement.
- **Variance Ratio (VR):** $VR = Var(y_{pred}) / Var(y_{true})$. Detects underdispersion where $VR < 1$.
- **KGE (Kling-Gupta Efficiency):** Evaluates overall agreement with three decomposed diagnostic subcomponents:
  - **$r$ (Pearson correlation):** Measures shape/timing dynamics.
  - **$\alpha$ (Variability ratio):** Measures amplitude dynamics (analogous to dispersion).
  - **$\beta$ (Bias ratio):** Measures systematic volumetric shifts.
- **Skill_VP (Variance-Penalised Skill):** An auxiliary framework combining skill and variability: $Skill\_VP = Skill\_RMSE \times \min(1, \alpha^2)$.

## Usage

You can reproduce the unified foundational dataset and run the initial persistence baseline experiment by using the following commands:

```bash
# 1. Unify and standardize the raw CSV data
python3 src/preprocessing/build_unified_pm10_dataset.py

# 2. Run the naive tracking baseline and calculate metric tables
python3 experiments/run_paper_c_baseline.py
```

## Output Files

Executing the baseline script will automatically generate the following diagnostic artifacts in the `results/` directory:
- `results/metrics_baseline_by_horizon.csv`: Tabular overview mapping all metrics (RMSE, Skill, VR, KGE components) per station and horizon.
- `results/predictions_baseline.csv`: Raw out-of-sample persistence predictions logged alongside the true ground values.

## LightGBM Capacity Robustness Experiment

This experiment evaluates whether the RMSE/structural-fidelity decoupling pattern
changes as LightGBM model capacity is varied.  Three fixed configurations are
compared (`lgbm_small`, `lgbm_base`, `lgbm_large`) under the same rolling-origin
protocol, using identical features, lags, and dataset.

### Running the sweep

```bash
# Step 1: run the three-configuration sweep
python experiments/run_paper_c_lightgbm_capacity_sweep.py

# Step 2: generate the summary table, findings text, and figure
python experiments/build_paper_c_lightgbm_capacity_summary.py
```

### Generated files

| File | Description |
|------|-------------|
| `results/metrics_lightgbm_capacity_sweep.csv` | Full metric table (station × model × horizon) |
| `results/paper_c_lightgbm_capacity_summary.csv` | Compact comparison focused on RMSE, Skill_RMSE, kge_alpha, VR, Skill_VP |
| `results/paper_c_lightgbm_capacity_findings.txt` | Auto-generated academic summary of capacity trade-off analysis |
| `results/fig_paper_c_lgbm_capacity_alpha_rmse.png` | Bar chart comparing RMSE and KGE alpha across configurations, stations, and horizons |
