import pandas as pd
from pathlib import Path
import os

def build_unified_dataset():
    base_dir = Path(__file__).parent.parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    files = [
        {"path": raw_dir / "PM10_Madrid_60_hourly.csv", "city": "Madrid", "station": "Madrid_60"},
        {"path": raw_dir / "PM10_Valencia_Politecnico_hourly.csv", "city": "Valencia", "station": "Valencia_Politecnico"}
    ]
    
    dfs = []
    for info in files:
        if not info["path"].exists():
            print(f"Warning: {info['path']} missing.")
            continue
            
        df = pd.read_csv(info["path"])
        df = df.rename(columns={"datetime": "timestamp", "value": "pm10"})
        df["station"] = info["station"]
        df["city"] = info["city"]
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["pm10"] = pd.to_numeric(df["pm10"], errors="coerce")
        df = df.dropna(subset=["pm10", "timestamp"])
        
        # In this dataset, there might be values <= 0 being invalid if PM10 is strictly positive in nature,
        # but the prompt specifically asked for "elimina filas inválidas", mainly nans.
        # We also sort temporally.
        df = df.sort_values("timestamp")
        dfs.append(df)
        
    if not dfs:
        print("No data found to process.")
        return
        
    unified_df = pd.concat(dfs, ignore_index=True)
    unified_df = unified_df[["timestamp", "pm10", "station", "city"]]
    
    output_path = processed_dir / "pm10_two_station_hourly.csv"
    unified_df.to_csv(output_path, index=False)
    print(f"Unified dataset saved to {output_path} with {len(unified_df)} rows.")

if __name__ == "__main__":
    build_unified_dataset()
