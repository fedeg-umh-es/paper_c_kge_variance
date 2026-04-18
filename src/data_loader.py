"""EEA Air Quality data download and loading utilities."""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

EEA_EXPORT_URL = (
    "https://fme.discomap.eea.europa.eu/fmedatastreaming/AirQualityDownload/"
    "AQData_Extract.fmw"
)

COUNTRY_CODE = "ES"
POLLUTANT_NOTATION = "PM10"
# EuroAirQuality component ID for PM10
PM10_COMPONENT_ID = 5


def download_eea_data(
    station_codes: list[str],
    pollutant: str = "PM10",
    year_start: int = 2019,
    year_end: int = 2023,
) -> pd.DataFrame:
    """Download hourly PM10 data for given EEA station codes.

    Fetches data from the EEA AirQuality FME export service, saves each
    station-year combination as a CSV in data/raw/, and returns a combined
    DataFrame.

    Parameters
    ----------
    station_codes:
        EEA local station codes (e.g. ['ES1422A', 'ES1938A']).
    pollutant:
        Pollutant notation string (only 'PM10' is supported currently).
    year_start, year_end:
        Inclusive year range.

    Returns
    -------
    pd.DataFrame
        Columns: datetime (UTC, tz-aware), station_code, pm10_value.
        Values < 0 or > 1000 are replaced with NaN without imputation.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    for station in station_codes:
        for year in range(year_start, year_end + 1):
            out_path = RAW_DIR / f"{station}_{year}.csv"
            if out_path.exists():
                df = pd.read_csv(out_path, parse_dates=["datetime"])
                frames.append(df)
                continue

            params = {
                "CountryCode": COUNTRY_CODE,
                "CityName": "",
                "Pollutant": PM10_COMPONENT_ID,
                "Year_from": year,
                "Year_to": year,
                "Station": station,
                "Samplingpoint": "",
                "Source": "E1a",
                "Output": "TEXT",
                "UpdateDate": "",
                "TimeCoverage": "Year",
            }

            try:
                resp = requests.get(EEA_EXPORT_URL, params=params, timeout=120)
                resp.raise_for_status()
            except requests.RequestException as exc:
                print(f"[WARN] Could not download {station}/{year}: {exc}")
                continue

            lines = resp.text.strip().splitlines()
            if len(lines) < 2:
                print(f"[WARN] Empty response for {station}/{year}")
                continue

            from io import StringIO
            raw_df = pd.read_csv(StringIO(resp.text))

            raw_df = _parse_eea_response(raw_df, station)
            raw_df.to_csv(out_path, index=False)
            frames.append(raw_df)

    if not frames:
        return pd.DataFrame(columns=["datetime", "station_code", "pm10_value"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime", "station_code"])
    combined = combined.sort_values(["station_code", "datetime"]).reset_index(drop=True)

    # Mark invalid values without imputing
    mask = (combined["pm10_value"] < 0) | (combined["pm10_value"] > 1000)
    combined.loc[mask, "pm10_value"] = np.nan

    return combined


def _parse_eea_response(raw_df: pd.DataFrame, station_code: str) -> pd.DataFrame:
    """Normalise an EEA CSV response to the standard schema."""
    # EEA column names vary; try common patterns
    time_col = next(
        (c for c in raw_df.columns if "start" in c.lower() or "date" in c.lower()),
        raw_df.columns[0],
    )
    value_col = next(
        (c for c in raw_df.columns if "value" in c.lower() or "concentration" in c.lower()),
        raw_df.columns[-1],
    )

    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(raw_df[time_col], utc=True, errors="coerce"),
            "station_code": station_code,
            "pm10_value": pd.to_numeric(raw_df[value_col], errors="coerce"),
        }
    )
    return df.dropna(subset=["datetime"])


def load_processed_data(station_code: str) -> pd.Series:
    """Load a cleaned hourly PM10 series from data/processed/.

    Parameters
    ----------
    station_code:
        EEA station identifier.

    Returns
    -------
    pd.Series
        Hourly PM10 values indexed by DatetimeIndex (UTC).
    """
    path = PROCESSED_DIR / f"{station_code}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found for {station_code}. "
            "Run data_loader.download_eea_data() and preprocessing.clean_series() first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    series = df.squeeze()
    series.name = station_code
    series.index.name = "datetime"
    return series
