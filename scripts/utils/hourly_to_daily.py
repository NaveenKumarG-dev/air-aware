#!/usr/bin/env python3
"""
Utility: Process Hourly Data
Convert hourly data to daily aggregates for forecasting
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

HOURLY_PATH = "data/raw/city_hour.csv"
OUTPUT_PATH = "data/processed/city_hour_daily.csv"

# Create directories if they don't exist
os.makedirs("data/processed", exist_ok=True)


def process_hourly():
    """Convert hourly data to daily aggregates"""
    print("Loading hourly data...")
    df = pd.read_csv(HOURLY_PATH)
    print(f"Loaded: {df.shape[0]} rows")

    # Parse datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Date"] = df["Datetime"].dt.date

    # Aggregate to daily
    pollutants = [
        "PM2.5",
        "PM10",
        "NO",
        "NO2",
        "NOx",
        "NH3",
        "CO",
        "SO2",
        "O3",
        "Benzene",
        "Toluene",
        "Xylene",
    ]

    daily = df.groupby(["City", "Date"])[pollutants].mean().reset_index()
    daily["Date"] = pd.to_datetime(daily["Date"])

    # Calculate AQI (simplified)
    daily["AQI"] = (
        daily["PM2.5"] * 0.5
        + daily["PM10"] * 0.3
        + daily["NO2"] * 0.1
        + daily["CO"] * 0.05
        + daily["SO2"] * 0.05
    )

    # Save
    daily.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved daily aggregates: {daily.shape[0]} rows")
    print(f"Date range: {daily['Date'].min()} to {daily['Date'].max()}")
    print(f"Cities: {daily['City'].nunique()}")


if __name__ == "__main__":
    process_hourly()
