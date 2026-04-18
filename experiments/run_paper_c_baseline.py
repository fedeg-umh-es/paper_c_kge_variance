import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.rolling_origin import generate_folds
from src.models.persistence import PersistenceModel
from src.evaluation.metrics import horizon_profile

def run_baseline_experiment():
    data_path = ROOT / "data" / "processed" / "pm10_two_station_hourly.csv"
    if not data_path.exists():
        print(f"Error: Processed data not found at {data_path}")
        return

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    
    horizons = [1, 6, 12, 24]
    n_folds = 8
    min_train_size = 2160
    
    all_metrics = []
    all_preds = []
    
    for station, group in df.groupby("station"):
        print(f"\nProcessing station: {station}")
        group = group.sort_values("timestamp").reset_index(drop=True)
        series = group["pm10"]
        values = series.values.astype(float)
        
        # We need to collect predictions for later saving as well
        accum = {h: {"y_true": [], "y_pred": [], "y_pers": [], "timestamp": []} for h in horizons}
        
        # We only have persistence model as the baseline here!
        for fold in generate_folds(series, n_folds=n_folds, min_train_size=min_train_size, horizons=horizons):
            origin_pos = fold["origin_pos"]
            origin_val = values[origin_pos]
            test_idx = fold["test_idx"]
            
            for h in horizons:
                positions = test_idx[h]
                if len(positions) == 0:
                    continue
                pos = positions[0]
                
                y_true_val = values[pos]
                y_pers_val = origin_val # Persistence prediction
                ts = group.loc[pos, "timestamp"]
                
                accum[h]["y_true"].append(y_true_val)
                accum[h]["y_pred"].append(y_pers_val) # Baseline uses persistence
                accum[h]["y_pers"].append(y_pers_val)
                accum[h]["timestamp"].append(ts)
                
        # Calculate skill metrics
        y_true_dict = {h: np.array(v["y_true"]) for h, v in accum.items()}
        y_pred_dict = {h: np.array(v["y_pred"]) for h, v in accum.items()}
        y_pers_dict = {h: np.array(v["y_pers"]) for h, v in accum.items()}
        
        profile = horizon_profile(y_true_dict, y_pred_dict, y_pers_dict, horizons)
        profile.insert(0, "station", station)
        all_metrics.append(profile)
        
        for h in horizons:
            h_df = pd.DataFrame({
                "station": station,
                "horizon": h,
                "timestamp": accum[h]["timestamp"],
                "y_true": accum[h]["y_true"],
                "prediction_persistence": accum[h]["y_pred"]
            })
            all_preds.append(h_df)
            
    if all_metrics:
        results_dir = ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_df = pd.concat(all_metrics, ignore_index=True)
        metrics_out = results_dir / "metrics_baseline_by_horizon.csv"
        metrics_df.to_csv(metrics_out, index=False)
        print(f"Metrics saved to {metrics_out}")
        
        preds_df = pd.concat(all_preds, ignore_index=True)
        preds_out = results_dir / "predictions_baseline.csv"
        preds_df.to_csv(preds_out, index=False)
        print(f"Predictions saved to {preds_out}")
        
        print("\n--- Summary ---")
        print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    run_baseline_experiment()
