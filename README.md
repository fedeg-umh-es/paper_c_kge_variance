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

You can reproduce the unified foundational dataset and run the baseline experiments by using the following commands:

```bash
# 1. Unify and standardize the raw CSV data
python3 src/preprocessing/build_unified_pm10_dataset.py

# 2. Run the naive tracking baseline and calculate metric tables
python3 experiments/run_paper_c_baseline.py

# 3. Run the rolling-origin ARIMA baseline (directly comparable to persistence)
python3 experiments/run_paper_c_arima.py

# 4. Build the ARIMA-vs-persistence summary table and automatic findings
python3 experiments/build_paper_c_arima_summary.py

# 5. Run the rolling-origin LightGBM experiment (directly comparable to persistence and ARIMA)
python3 experiments/run_paper_c_lightgbm.py

# 6. Build the LightGBM-vs-persistence summary table and automatic findings
python3 experiments/build_paper_c_lightgbm_summary.py

# 7. Build the master three-way comparison package (Paper C final outputs)
python3 experiments/build_paper_c_master_comparison.py
```

### Paper C final outputs

`build_paper_c_master_comparison.py` is the single entry point for the
results section. It requires steps 1–6 to have been run first and
produces:

| File | Use |
|------|-----|
| `results/paper_c_table_core.csv` | **Paper body** — compact table with RMSE, Skill, alpha, VR, KGE, Skill_VP per model and horizon |
| `results/paper_c_table_appendix.csv` | **Appendix** — full wide table with all metrics and derived diagnostic flags |
| `results/paper_c_master_comparison.csv` | Complete merged dataset with all flags (machine-readable) |
| `results/paper_c_main_findings.txt` | Auto-generated narrative summary (global counts, per-station breakdown, ARIMA vs LightGBM section) |
| `results/fig_paper_c_rmse_skill.png` | RMSE-by-horizon line chart per station (three models) |
| `results/fig_paper_c_alpha_vr.png` | KGE alpha-by-horizon chart per station (ARIMA vs LightGBM, with collapse thresholds) |

The ARIMA experiment uses the same rolling-origin protocol as the persistence
baseline (identical folds, identical train-only logic, identical horizons
`[1, 6, 12, 24]`). A fixed, conservative configuration `ARIMA(2, 1, 2)` is
fitted once per station and per fold on the corresponding training window
only — no seasonal terms, no auto-search — and forecasts are produced for
each required horizon. The persistence baseline remains available and
unchanged; skill scores are still referenced against it.

The LightGBM experiment follows the identical protocol. One regressor is
trained per horizon and per fold on autoregressive lag features of PM10
(lags 1, 2, 3, 6, 12, 24). A direct multi-step strategy is used (one
model per horizon, no recursive chaining). Hyperparameters are fixed and
conservative (`n_estimators=300`, `num_leaves=31`, `max_depth=5`,
`learning_rate=0.05`) — no grid search, no auto-ML. Any per-fold failure
falls back to persistence with a warning and does not abort execution.

## Output Files

Executing the baseline scripts will automatically generate the following diagnostic artifacts in the `results/` directory:
- `results/metrics_baseline_by_horizon.csv`: Tabular overview mapping all metrics (RMSE, Skill, VR, KGE components) per station and horizon for the persistence baseline.
- `results/predictions_baseline.csv`: Raw out-of-sample persistence predictions logged alongside the true ground values.
- `results/metrics_arima_by_horizon.csv`: Same metric schema as the persistence baseline, computed for the rolling-origin ARIMA model and therefore directly comparable.
- `results/predictions_arima.csv`: Per-fold ARIMA point forecasts with the matching persistence forecast and true value on each row.
- `results/paper_c_arima_vs_persistence_summary.csv`: Wide-format side-by-side table merging the persistence and ARIMA metrics by `(station, horizon)` with derived flags (`delta_rmse`, `better_than_persistence`, `variance_collapse_flag`, `severe_variance_collapse_flag`, `overdispersion_flag`).
- `results/paper_c_arima_key_findings.txt`: Short auto-generated per-station reading highlighting where ARIMA improves RMSE, where variance collapses, where both happen simultaneously, and where ARIMA adds no skill.
- `results/metrics_lightgbm_by_horizon.csv`: Same metric schema, computed for the rolling-origin LightGBM model.
- `results/predictions_lightgbm.csv`: Per-fold LightGBM point forecasts with the matching persistence forecast and true value.
- `results/paper_c_lightgbm_vs_persistence_summary.csv`: Wide-format side-by-side table merging persistence and LightGBM metrics by `(station, horizon)` with derived diagnostic flags.
- `results/paper_c_lightgbm_key_findings.txt`: Auto-generated per-station reading for the LightGBM experiment.
- `results/paper_c_master_comparison.csv`: Full three-way merged dataset (all metrics + all derived flags).
- `results/paper_c_table_core.csv`: Compact table for the **paper body** — use this for the Results section.
- `results/paper_c_table_appendix.csv`: Extended table for the **appendix** — all metrics and flags.
- `results/paper_c_main_findings.txt`: Auto-generated narrative covering global counts, per-station breakdowns, and ARIMA vs LightGBM comparison.
- `results/fig_paper_c_rmse_skill.png`: RMSE comparison figure (three models × all horizons × all stations).
- `results/fig_paper_c_alpha_vr.png`: KGE alpha comparison figure with collapse threshold lines.
- `results/metrics_lightgbm_by_horizon.csv`: Same metric schema, computed for the rolling-origin LightGBM model.
- `results/predictions_lightgbm.csv`: Per-fold LightGBM point forecasts with the matching persistence forecast and true value.
- `results/paper_c_lightgbm_vs_persistence_summary.csv`: Wide-format side-by-side table merging persistence and LightGBM metrics by `(station, horizon)` with derived diagnostic flags.
- `results/paper_c_lightgbm_key_findings.txt`: Auto-generated per-station reading for the LightGBM experiment.
