# Air Quality Forecasting System

A time series forecasting system to predict Air Quality Index (AQI) and pollutant levels using historical data from Indian cities.

## Problem Statement

Air pollution is a growing concern in urban environments, directly impacting human health and environmental quality. Accurate forecasting of air quality can help authorities take preventive measures, issue warnings, and inform the public.

This project builds a forecasting system that:
- Predicts AQI and key pollutants (PM2.5, PM10, NO2, etc.) using machine learning
- Provides visual dashboard showing historical trends and future predictions
- Issues alerts when predicted AQI crosses safety thresholds
- Analyzes pollutant contributions and seasonal trends

## Dataset

**Source**: Central Pollution Control Board (CPCB) India - [Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

| File | Records | Description |
|------|---------|-------------|
| `city_day.csv` | 29,531 | Daily readings by city (2015-2020) |
| `city_hour.csv` | ~630,000 | Hourly readings by city |
| `station_day.csv` | ~175,000 | Daily readings by station |

**Fields in city_day.csv:**
- `City`, `Date`: Location and timestamp
- Pollutants: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- `AQI`, `AQI_Bucket`: Target variable and categories

**AQI Categories:**
| AQI Range | Category | Color |
|-----------|----------|-------|
| 0-50 | Good | Green |
| 51-100 | Satisfactory | Light Green |
| 101-200 | Moderate | Yellow |
| 201-300 | Poor | Orange |
| 301-400 | Very Poor | Red |
| 401+ | Severe | Maroon |

---

## ⚠️ Important: Dataset Files

Due to GitHub's file size limits, some large files are not included in this repository.

### Files Included (in `data/raw/`)
- `city_day.csv` ✅
- `station_day.csv` ✅
- `stations.csv` ✅

### Files to Download Separately
The following files are **excluded** due to size. Download from Kaggle:

| File | Size | Download Link |
|------|------|---------------|
| `city_hour.csv` | ~63 MB | [Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) |
| `station_hour.csv` | ~210 MB | [Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) |

### How to Download
1. Go to: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
2. Download the dataset
3. Extract and place `city_hour.csv` and `station_hour.csv` in `data/raw/`

---

## Quick Start

```bash
# Run everything in one command
uv run python run_all.py

# Launch dashboard
uv run streamlit run app.py

## Project Structure

```
air-quality-prediction/
├── data/
│   ├── raw/                       # Source data (add your CSV files here)
│   │   ├── city_day.csv           # Primary dataset
│   │   ├── city_hour.csv         # Optional: hourly data
│   │   ├── station_day.csv       # Station-level daily
│   │   ├── station_hour.csv      # Station-level hourly
│   │   └── stations.csv          # Station metadata
│   └── processed/                 # Generated data (auto-created)
│       ├── preprocessed_city_day.csv
│       └── features_city_day.csv
├── scripts/
│   ├── utils/
│   │   └── hourly_to_daily.py   # Optional: convert hourly→daily
│   ├── 01_exploration.py         # Phase 1: EDA
│   ├── 02_preprocessing.py       # Phase 2: Data cleaning
│   ├── 03_features.py            # Phase 3: Feature engineering
│   └── 04_modeling.py           # Phase 4: Model training
├── models/                       # Trained models (auto-created)
├── run_all.py                   # Run all phases
├── app.py                       # Streamlit dashboard
├── pyproject.toml               # Dependencies
└── README.md
```

## Folder Details

| Folder | Purpose | Auto-created? |
|--------|---------|----------------|
| `data/raw/` | Source CSV files | No (you add them) |
| `data/processed/` | Cleaned & engineered data | ✅ Yes |
| `models/` | Trained `.pkl` files | ✅ Yes |

## Run Individual Phases

```bash
# Phase 1: Exploration
uv run python scripts/01_exploration.py

# Phase 2: Preprocessing
uv run python scripts/02_preprocessing.py

# Phase 3: Features
uv run python scripts/03_features.py

# Phase 4: Modeling
uv run python scripts/04_modeling.py
```

## Scripts Explained

| Script | What it does |
|--------|--------------|
| `hourly_to_daily.py` | Optional: Convert hourly CSV to daily aggregates |
| `01_exploration.py` | Load data, analyze distributions, print stats |
| `02_preprocessing.py` | Clean data, handle missing, filter cities |
| `03_features.py` | Create lag features, date features, rolling stats |
| `04_modeling.py` | Train ARIMA/XGBoost/Prophet/LSTM, compare & save |

## Key Files

| File | Description |
|------|-------------|
| `data/raw/city_day.csv` | **Your source data** - download from Kaggle |
| `data/processed/preprocessed_city_day.csv` | Cleaned data after Phase 2 |
| `data/processed/features_city_day.csv` | Data with features after Phase 3 |
| `models/xgboost_model.pkl` | Best performing model (XGBoost) |
| `run_all.py` | One command to run all phases |

## Expected Output

After running `run_all.py`:
- `data/processed/` - 2 CSV files (preprocessed, features)
- `models/` - 7 files (models + encoders + results)

---

## What Was Done (Phases)

### Phase 1: Data Exploration (`scripts/01_exploration.py`)
Analyzed the dataset to understand its structure:
- **Records**: 29,531 daily readings
- **Cities**: 26 unique cities across India
- **Date Range**: January 2015 - July 2020 (5.5 years)
- **Missing Values**: Various pollutants had missing data (PM2.5: 15.6%, PM10: 37.7%, etc.)
- **AQI Statistics**: Mean 166, Max 2049
- **Key Insight**: PM10 (0.80), CO (0.68), and PM2.5 (0.66) have highest correlation with AQI

### Phase 2: Data Preprocessing (`scripts/02_preprocessing.py`)
Cleaned and prepared data for modeling:
- Converted Date column to datetime
- Filtered to cities with ≥500 days of data (19 cities retained)
- Handled missing values using linear interpolation + median fill
- Capped outliers at ±4 standard deviations
- Created AQI bucket categories

**Output**: `data/processed/preprocessed_city_day.csv` (27,805 rows, 0 missing values)

### Phase 3: Feature Engineering (`scripts/03_features.py`)
Created features for time series forecasting:
- **Date Features**: Year, Month, Day, DayOfWeek, Quarter, Season, IsWeekend
- **Lag Features**: AQI_lag_1, AQI_lag_2, AQI_lag_3, AQI_lag_7, AQI_lag_14, AQI_lag_30
- **Rolling Statistics**: 7-day, 14-day, 30-day rolling mean and std
- **Pollutant Lags**: PM2.5_lag_1, PM10_lag_1, etc.
- **Expanding Mean**: Cumulative average
- **Categorical Encoding**: Season and City encoded numerically

**Output**: `data/processed/features_city_day.csv` (52 features)

### Phase 4: Model Training (`scripts/04_modeling.py`)
Trained and compared multiple models:

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **XGBoost** | **14.24** | **20.96** | **0.97** |
| LSTM | 36.65 | 49.24 | - |
| Prophet | 63.45 | 81.29 | - |
| ARIMA | 90.40 | 114.04 | - |

**Best Model**: XGBoost

**Pollutant-Specific Models**:
| Pollutant | Model | MAE | RMSE | R² |
|-----------|-------|-----|------|-----|
| PM2.5 | XGBoost | 21.67 | 31.67 | 0.81 |
| PM10 | XGBoost | 41.80 | 55.17 | 0.69 |

**Top Features**:
1. PM2.5 (53.7%)
2. AQI_lag_1 (23.2%)
3. PM10 (6.8%)
4. NOx (2.5%)
5. CO (2.3%)

**Output**: Trained models saved to `models/`

### Phase 5: Dashboard (`app.py`)
Streamlit web application with:
- City selector dropdown
- Current AQI display with gauge
- Alert banners (Good/Moderate/Poor/Severe)
- Historical trend charts (30-365 days)
- 7-day forecast
- Pollutant analysis (bar charts, distributions)
- AQI category pie chart
- Admin panel for model retraining and data upload

---

## Results Summary

| Metric | Value |
|--------|-------|
| Best Model | XGBoost |
| MAE | 14.24 |
| RMSE | 20.96 |
| R² Score | 0.97 |
| Cities Covered | 19 |
| Date Range | 2015-2020 |
| Pollutants Modeled | AQI, PM2.5, PM10 |

---

## Installation & Usage

### 1. Install Dependencies
```bash
uv add pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels plotly streamlit prophet tensorflow
```

### 2. Run All Phases
```bash
uv run python run_all.py
```

### 3. Launch Dashboard
```bash
uv run streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## Requirements

- Python 3.8+
- pandas, numpy
- matplotlib / seaborn / plotly
- scikit-learn, xgboost
- statsmodels, prophet, tensorflow
- streamlit

---

## Future Improvements

- Implement per-city models for better regional accuracy
- Add more sophisticated forecasting (Transformer models)
- Enhance dashboard with more interactive features
- Include meteorological features (temperature, humidity) if available
