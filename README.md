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
```

The ARIMA experiment uses the same rolling-origin protocol as the persistence
baseline (identical folds, identical train-only logic, identical horizons
`[1, 6, 12, 24]`). A fixed, conservative configuration `ARIMA(2, 1, 2)` is
fitted once per station and per fold on the corresponding training window
only — no seasonal terms, no auto-search — and forecasts are produced for
each required horizon. The persistence baseline remains available and
unchanged; skill scores are still referenced against it.

## Output Files

Executing the baseline scripts will automatically generate the following diagnostic artifacts in the `results/` directory:
- `results/metrics_baseline_by_horizon.csv`: Tabular overview mapping all metrics (RMSE, Skill, VR, KGE components) per station and horizon for the persistence baseline.
- `results/predictions_baseline.csv`: Raw out-of-sample persistence predictions logged alongside the true ground values.
- `results/metrics_arima_by_horizon.csv`: Same metric schema as the persistence baseline, computed for the rolling-origin ARIMA model and therefore directly comparable.
- `results/predictions_arima.csv`: Per-fold ARIMA point forecasts with the matching persistence forecast and true value on each row.
- `results/paper_c_arima_vs_persistence_summary.csv`: Wide-format side-by-side table merging the persistence and ARIMA metrics by `(station, horizon)` with derived flags (`delta_rmse`, `better_than_persistence`, `variance_collapse_flag`, `severe_variance_collapse_flag`, `overdispersion_flag`).
- `results/paper_c_arima_key_findings.txt`: Short auto-generated per-station reading highlighting where ARIMA improves RMSE, where variance collapses, where both happen simultaneously, and where ARIMA adds no skill.
