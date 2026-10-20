#!/usr/bin/env python3
"""
Phase 1: Data Exploration
Analyze city_day.csv - Daily Air Quality Index data for Indian cities

Note: Visualizations are available in the Streamlit dashboard (app.py)
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/city_day.csv"


def load_data():
    """Load the dataset"""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_basic(df):
    """Basic exploration"""
    print("\n" + "=" * 50)
    print("BASIC INFO")
    print("=" * 50)
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")


def explore_missing(df):
    """Check missing values"""
    print("\n" + "=" * 50)
    print("MISSING VALUES")
    print("=" * 50)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing": missing, "Percent": missing_pct})
    print(f"\n{missing_df[missing_df['Missing'] > 0]}")


def explore_cities(df):
    """Explore cities and date range"""
    print("\n" + "=" * 50)
    print("CITIES AND DATE RANGE")
    print("=" * 50)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"\nUnique cities: {df['City'].nunique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nCities: {sorted(df['City'].unique())}")


def explore_aqi(df):
    """Analyze AQI distribution"""
    print("\n" + "=" * 50)
    print("AQI ANALYSIS")
    print("=" * 50)
    print(f"\nAQI Statistics:\n{df['AQI'].describe()}")
    print(f"\nAQI Bucket distribution:\n{df['AQI_Bucket'].value_counts()}")


def explore_pollutants(df):
    """Analyze pollutant correlations"""
    print("\n" + "=" * 50)
    print("POLLUTANT CORRELATIONS WITH AQI")
    print("=" * 50)
    pollutants = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "NH3"]
    corr = (
        df[pollutants + ["AQI"]].corr()["AQI"].drop("AQI").sort_values(ascending=False)
    )
    print(f"\n{corr}")


def main():
    print("=" * 60)
    print("PHASE 1: DATA EXPLORATION")
    print("=" * 60)

    df = load_data()
    explore_basic(df)
    explore_missing(df)
    explore_cities(df)
    explore_aqi(df)
    explore_pollutants(df)

    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)
    print("\nNote: Visualizations available in Streamlit dashboard:")
    print("  uv run streamlit run app.py")


if __name__ == "__main__":
    main()
