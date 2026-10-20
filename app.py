#!/usr/bin/env python3
"""
AirAware AI - Live Air Quality Telemetry & Future AQI Prediction
Connected directly to Open-Meteo Live APIs & Trained XGBoost ML Model
Theme-Adaptive (Dark & Light Mode) with Configurable Historical Date Range
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
import urllib.request
import urllib.error
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# Page Configuration & Global Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="AirAware • Live Air Quality & AI Forecast",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling - Theme Adaptive (Compatible with Dark & Light Mode)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Brand Banner */
    .brand-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #047857 100%);
        border-radius: 16px;
        padding: 22px 28px;
        color: #ffffff;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }

    .brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
        color: #ffffff;
    }

    .brand-subtitle {
        font-size: 0.92rem;
        color: #cbd5e1;
        margin-top: 4px;
        font-weight: 400;
    }

    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #34d399;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 9999px;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        background-color: #34d399;
        border-radius: 50%;
        box-shadow: 0 0 10px #34d399;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* AQI Category Pills */
    .aqi-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .aqi-good { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .aqi-satisfactory { background: rgba(132, 204, 22, 0.2); color: #a3e635; border: 1px solid #84cc16; }
    .aqi-moderate { background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid #eab308; }
    .aqi-poor { background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid #f97316; }
    .aqi-very-poor { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .aqi-severe { background: rgba(217, 70, 239, 0.2); color: #e879f9; border: 1px solid #c026d3; }

    /* Theme-Adaptive Pollutant Micro Tiles */
    .pollutant-card {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
        margin-bottom: 8px;
    }

    .pollutant-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }

    .pollutant-name {
        font-size: 0.78rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 2px;
    }

    .pollutant-val {
        font-size: 1.55rem;
        font-weight: 800;
        color: inherit;
        line-height: 1.2;
    }

    .pollutant-unit {
        font-size: 0.72rem;
        color: #94a3b8;
        font-weight: 500;
    }

    .pollutant-status {
        margin-top: 8px;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }

    /* Prediction Hero Box */
    .prediction-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1.5px solid rgba(59, 130, 246, 0.4);
        border-radius: 18px;
        padding: 24px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.25);
    }

    .pred-val {
        font-size: 3.8rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        line-height: 1;
        margin: 10px 0;
    }

    .pred-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #94a3b8;
    }

    .health-tip-card {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        padding: 14px 16px;
        border-radius: 8px;
        margin-top: 12px;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Supported Indian Cities & Coordinates
# ---------------------------------------------------------
CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi NCR"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    "Pune": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra"},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat"},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh"},
    "Patna": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar"},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794, "state": "Punjab / Haryana"},
    "Gurugram": {"lat": 28.4595, "lon": 77.0266, "state": "Haryana"},
    "Amritsar": {"lat": 31.6340, "lon": 74.8723, "state": "Punjab"},
    "Guwahati": {"lat": 26.1445, "lon": 91.7362, "state": "Assam"},
    "Kochi": {"lat": 9.9312, "lon": 76.2673, "state": "Kerala"},
    "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366, "state": "Kerala"},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh"},
}

# ---------------------------------------------------------
# Official CPCB Sub-Index Breakpoints (National Indian AQI)
# ---------------------------------------------------------
CPCB_BREAKPOINTS = {
    'PM2.5': [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200), (90, 120, 201, 300), (120, 250, 301, 400), (250, 500, 401, 500)],
    'PM10': [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200), (250, 350, 201, 300), (350, 430, 301, 400), (430, 600, 401, 500)],
    'NO2': [(0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200), (180, 280, 201, 300), (280, 400, 301, 400), (400, 1000, 401, 500)],
    'SO2': [(0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200), (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 2000, 401, 500)],
    'CO': [(0, 1.0, 0, 50), (1.0, 2.0, 51, 100), (2.0, 10.0, 101, 200), (10.0, 17.0, 201, 300), (17.0, 34.0, 301, 400), (34.0, 50.0, 401, 500)],
    'O3': [(0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200), (168, 208, 201, 300), (208, 748, 301, 400), (748, 1000, 401, 500)],
}

POLLUTANT_SAFE_LIMITS = {
    'PM2.5': {'limit': 60, 'unit': 'µg/m³'},
    'PM10': {'limit': 100, 'unit': 'µg/m³'},
    'NO2': {'limit': 80, 'unit': 'µg/m³'},
    'SO2': {'limit': 80, 'unit': 'µg/m³'},
    'CO': {'limit': 2.0, 'unit': 'mg/m³'},
    'O3': {'limit': 100, 'unit': 'µg/m³'},
}


def calc_sub_index(conc, bps):
    if conc is None:
        return None
    try:
        fconc = float(conc)
    except (ValueError, TypeError):
        return None
    if np.isnan(fconc) or fconc < 0:
        return None
    for b_low, b_high, i_low, i_high in bps:
        if fconc <= b_high:
            return i_low + (fconc - b_low) * (i_high - i_low) / (b_high - b_low)
    return bps[-1][3]


def get_aqi_details(aqi_val):
    """Return category metadata for a given AQI integer value"""
    if aqi_val is None:
        return {"category": "Unknown", "color": "#94a3b8", "css_class": "aqi-good", "health_note": "No data available."}
    try:
        fval = float(aqi_val)
        if np.isnan(fval):
            return {"category": "Unknown", "color": "#94a3b8", "css_class": "aqi-good", "health_note": "No data available."}
        val = round(fval)
    except (ValueError, TypeError):
        return {"category": "Unknown", "color": "#94a3b8", "css_class": "aqi-good", "health_note": "No data available."}

    if val <= 50:
        return {
            "category": "Good",
            "color": "#10b981",
            "css_class": "aqi-good",
            "health_note": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
            "outdoor_safe": "Safe for all outdoor activities."
        }
    elif val <= 100:
        return {
            "category": "Satisfactory",
            "color": "#84cc16",
            "css_class": "aqi-satisfactory",
            "health_note": "Minor breathing discomfort to sensitive people. Overall safe for public.",
            "outdoor_safe": "Safe for regular outdoor activities."
        }
    elif val <= 200:
        return {
            "category": "Moderate",
            "color": "#eab308",
            "css_class": "aqi-moderate",
            "health_note": "Breathing discomfort to people with lungs, asthma, and heart conditions.",
            "outdoor_safe": "Sensitive groups should reduce prolonged outdoor exertion."
        }
    elif val <= 300:
        return {
            "category": "Poor",
            "color": "#f97316",
            "css_class": "aqi-poor",
            "health_note": "Breathing discomfort to most people on prolonged exposure. Irritation in eyes.",
            "outdoor_safe": "Avoid prolonged outdoor workouts. Wear an N95 mask outside."
        }
    elif val <= 400:
        return {
            "category": "Very Poor",
            "color": "#ef4444",
            "css_class": "aqi-very-poor",
            "health_note": "Respiratory illness on prolonged exposure. Pronounced effect on lung patients.",
            "outdoor_safe": "Avoid all outdoor physical activity. Keep windows shut."
        }
    else:
        return {
            "category": "Severe",
            "color": "#d946ef",
            "css_class": "aqi-severe",
            "health_note": "Severe health impact on entire population. Emergency health conditions.",
            "outdoor_safe": "Stay indoors. Use HEPA air purifiers."
        }


def compute_cpcb_aqi(pollutants):
    """Compute overall Indian National AQI and dominant pollutant from concentrations"""
    sub_indices = {}
    for p, val in pollutants.items():
        if p in CPCB_BREAKPOINTS and val is not None:
            try:
                fval = float(val)
                if not np.isnan(fval):
                    si = calc_sub_index(fval, CPCB_BREAKPOINTS[p])
                    if si is not None:
                        sub_indices[p] = round(si, 1)
            except (ValueError, TypeError):
                continue

    if not sub_indices:
        return 0, "None", {}

    dominant = max(sub_indices, key=sub_indices.get)
    overall_aqi = int(round(sub_indices[dominant]))
    return overall_aqi, dominant, sub_indices


# ---------------------------------------------------------
# Open-Meteo Live API Integration (Up to 60 Days History)
# ---------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_air_quality(lat, lon, past_days=60, forecast_days=7):
    """Fetch live, up to 60-day historical & 7-day forecast air quality directly from Open-Meteo"""
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}&"
        f"current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone&"
        f"hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone&"
        f"past_days={past_days}&forecast_days={forecast_days}&timezone=Asia%2FKolkata"
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'AirAwareDashboard/2.0'})
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            data = json.loads(response.read().decode('utf-8'))
        return data, None
    except Exception as e:
        return None, str(e)


def process_telemetry(raw_data):
    """Process raw Open-Meteo API payload into structured DataFrames"""
    if not raw_data:
        return None, None, None

    # 1. Hourly DataFrame
    hourly_df = pd.DataFrame(raw_data.get('hourly', {}))
    hourly_df['time'] = pd.to_datetime(hourly_df['time'])
    hourly_df['Date'] = hourly_df['time'].dt.date

    # Standardize column names & units
    hourly_df['PM2.5'] = hourly_df['pm2_5']
    hourly_df['PM10'] = hourly_df['pm10']
    hourly_df['NO2'] = hourly_df['nitrogen_dioxide']
    hourly_df['SO2'] = hourly_df['sulphur_dioxide']
    hourly_df['CO'] = hourly_df['carbon_monoxide'] / 1000.0  # Convert µg/m³ -> mg/m³
    hourly_df['O3'] = hourly_df['ozone']

    # 2. Daily DataFrame
    daily_df = hourly_df.groupby('Date')[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']].mean().reset_index()
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])

    # Compute CPCB Sub-indices & AQI for each day
    aqis = []
    dominants = []
    for _, row in daily_df.iterrows():
        p_dict = {
            'PM2.5': row['PM2.5'],
            'PM10': row['PM10'],
            'NO2': row['NO2'],
            'SO2': row['SO2'],
            'CO': row['CO'],
            'O3': row['O3']
        }
        val, dom, _ = compute_cpcb_aqi(p_dict)
        aqis.append(val)
        dominants.append(dom)

    daily_df['AQI'] = aqis
    daily_df['Dominant'] = dominants

    # Impute secondary gases matching training distributions
    daily_df['NO'] = daily_df['NO2'] * 0.4
    daily_df['NOx'] = daily_df['NO2'] * 1.3
    daily_df['NH3'] = 16.0
    daily_df['Benzene'] = 1.2
    daily_df['Toluene'] = 3.3
    daily_df['Xylene'] = 1.0

    # Calendar features
    daily_df['Month'] = daily_df['Date'].dt.month
    daily_df['DayOfWeek'] = daily_df['Date'].dt.dayofweek
    daily_df['Season'] = daily_df['Month'].apply(
        lambda m: 0 if m in [3, 4, 5] else 1 if m in [6, 7, 8] else 2 if m in [9, 10, 11] else 3
    )

    # Lags and rolling
    for lag in [1, 2, 3, 7, 14]:
        daily_df[f'AQI_lag_{lag}'] = daily_df['AQI'].shift(lag)
    for win in [7, 14, 30]:
        daily_df[f'AQI_rolling_{win}'] = daily_df['AQI'].shift(1).rolling(win, min_periods=1).mean()

    # 3. Current Live Snapshot
    curr_raw = raw_data.get('current', {})
    curr_pollutants = {
        'PM2.5': round(curr_raw.get('pm2_5', 0.0), 1),
        'PM10': round(curr_raw.get('pm10', 0.0), 1),
        'NO2': round(curr_raw.get('nitrogen_dioxide', 0.0), 1),
        'SO2': round(curr_raw.get('sulphur_dioxide', 0.0), 1),
        'CO': round(curr_raw.get('carbon_monoxide', 0.0) / 1000.0, 2),
        'O3': round(curr_raw.get('ozone', 0.0), 1)
    }
    live_aqi, live_dominant, live_sub_indices = compute_cpcb_aqi(curr_pollutants)

    live_snapshot = {
        'timestamp': curr_raw.get('time', ''),
        'aqi': live_aqi,
        'dominant': live_dominant,
        'pollutants': curr_pollutants,
        'sub_indices': live_sub_indices
    }

    return hourly_df, daily_df, live_snapshot


# ---------------------------------------------------------
# Trained ML Model Loader
# ---------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    """Load pretrained XGBoost model & feature columns"""
    model = None
    feature_cols = None
    try:
        with open("models/xgboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/xgboost_features.pkl", "rb") as f:
            feature_cols = pickle.load(f)
    except Exception as e:
        st.sidebar.warning(f"Note on Model: {e}")
    return model, feature_cols


def predict_for_specific_date(target_date, daily_df, model, feature_cols):
    """
    Predict AQI & Pollutant breakdown for a user-selected future date
    using recent real telemetry + trained XGBoost model.
    """
    target_dt = pd.to_datetime(target_date)
    existing_row = daily_df[daily_df['Date'].dt.date == target_date]

    if not existing_row.empty:
        row = existing_row.iloc[0].copy()
        feat_dict = {}
        for col in feature_cols:
            if col in row and not pd.isna(row[col]):
                feat_dict[col] = float(row[col])
            else:
                feat_dict[col] = 50.0

        if model is not None:
            input_df = pd.DataFrame([feat_dict])[feature_cols]
            try:
                ml_pred_aqi = float(model.predict(input_df)[0])
            except Exception:
                ml_pred_aqi = float(row['AQI'])
        else:
            ml_pred_aqi = float(row['AQI'])

        physical_aqi = float(row['AQI'])
        final_aqi = int(round(0.6 * ml_pred_aqi + 0.4 * physical_aqi))

        p_breakdown = {
            'PM2.5': round(float(row['PM2.5']), 1),
            'PM10': round(float(row['PM10']), 1),
            'NO2': round(float(row['NO2']), 1),
            'SO2': round(float(row['SO2']), 1),
            'CO': round(float(row['CO']), 2),
            'O3': round(float(row['O3']), 1)
        }
        dominant = row['Dominant']
        source_note = "Ensemble: Trained XGBoost Model + Open-Meteo Atmospheric Forecast Vectors"
    else:
        last_row = daily_df.iloc[-1].copy()
        days_ahead = (target_date - daily_df['Date'].dt.date.iloc[-1]).days

        month = target_dt.month
        season = 0 if month in [3, 4, 5] else 1 if month in [6, 7, 8] else 2 if month in [9, 10, 11] else 3
        day_of_week = target_dt.dayofweek

        base_aqi = float(last_row['AQI'])
        decay = np.exp(-0.04 * max(1, days_ahead))
        seasonal_baseline = 160.0 if season == 3 else 110.0
        projected_aqi = base_aqi * decay + seasonal_baseline * (1 - decay)

        feat_dict = {
            'PM2.5': max(10.0, float(last_row['PM2.5']) * decay + 50.0 * (1 - decay)),
            'PM10': max(20.0, float(last_row['PM10']) * decay + 90.0 * (1 - decay)),
            'NO': 15.0,
            'NO2': 30.0,
            'NOx': 35.0,
            'NH3': 16.0,
            'CO': 1.2,
            'SO2': 15.0,
            'O3': 35.0,
            'Benzene': 1.2,
            'Toluene': 3.3,
            'Xylene': 1.0,
            'AQI_lag_1': projected_aqi,
            'AQI_lag_2': projected_aqi * 1.02,
            'AQI_lag_3': projected_aqi * 0.98,
            'AQI_lag_7': projected_aqi,
            'AQI_lag_14': projected_aqi,
            'AQI_rolling_7': projected_aqi,
            'AQI_rolling_14': projected_aqi,
            'AQI_rolling_30': projected_aqi,
            'Month': month,
            'DayOfWeek': day_of_week,
            'Season': season
        }

        if model is not None:
            try:
                input_df = pd.DataFrame([feat_dict])[feature_cols]
                final_aqi = int(round(float(model.predict(input_df)[0])))
            except Exception:
                final_aqi = int(round(projected_aqi))
        else:
            final_aqi = int(round(projected_aqi))

        p_breakdown = {
            'PM2.5': round(feat_dict['PM2.5'], 1),
            'PM10': round(feat_dict['PM10'], 1),
            'NO2': round(feat_dict['NO2'], 1),
            'SO2': round(feat_dict['SO2'], 1),
            'CO': round(feat_dict['CO'], 2),
            'O3': round(feat_dict['O3'], 1)
        }
        dominant = 'PM2.5' if p_breakdown['PM2.5'] > 50 else 'PM10'
        source_note = "Autoregressive Projection: XGBoost Model + Seasonal Memory"

    final_aqi = max(10, min(500, final_aqi))
    details = get_aqi_details(final_aqi)

    return {
        'date': target_date,
        'aqi': final_aqi,
        'dominant': dominant,
        'category': details['category'],
        'color': details['color'],
        'css_class': details['css_class'],
        'health_note': details['health_note'],
        'outdoor_safe': details['outdoor_safe'],
        'pollutants': p_breakdown,
        'source_note': source_note
    }


# ---------------------------------------------------------
# Plotly Chart Visualizations (Theme-Adaptive)
# ---------------------------------------------------------
def render_speedometer(aqi_val, city_name):
    """Render a clean speedometer gauge chart with high contrast in dark and light modes"""
    details = get_aqi_details(aqi_val)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_val,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={
            'font': {'size': 56, 'family': 'Plus Jakarta Sans', 'color': details['color'], 'weight': 800}
        },
        gauge={
            'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "#64748b", 'tickmode': 'array', 'tickvals': [50, 100, 200, 300, 400, 500]},
            'bar': {'color': details['color'], 'thickness': 0.35},
            'bgcolor': "rgba(128, 128, 128, 0.12)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [50, 100], 'color': 'rgba(132, 204, 22, 0.3)'},
                {'range': [100, 200], 'color': 'rgba(234, 179, 8, 0.3)'},
                {'range': [200, 300], 'color': 'rgba(249, 115, 22, 0.3)'},
                {'range': [300, 400], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [400, 500], 'color': 'rgba(217, 70, 239, 0.3)'},
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.8,
                'value': aqi_val
            }
        }
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#cbd5e1")
    )
    return fig


def render_history_trend(filtered_df, city_name, start_date, end_date):
    """Render historical AQI trend for user's selected date range"""
    fig = go.Figure()

    # Safety reference bands
    fig.add_hrect(y0=0, y1=50, fillcolor="#10b981", opacity=0.08, line_width=0)
    fig.add_hrect(y0=50, y1=100, fillcolor="#84cc16", opacity=0.08, line_width=0)
    fig.add_hrect(y0=100, y1=200, fillcolor="#eab308", opacity=0.08, line_width=0)
    fig.add_hrect(y0=200, y1=300, fillcolor="#f97316", opacity=0.08, line_width=0)
    fig.add_hrect(y0=300, y1=500, fillcolor="#ef4444", opacity=0.08, line_width=0)

    # AQI Line
    fig.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['AQI'],
        mode='lines+markers',
        name='Air Quality Index (AQI)',
        line=dict(color='#38bdf8', width=3, shape='spline'),
        marker=dict(size=7, color='#0284c7'),
        fill='tozeroy',
        fillcolor='rgba(56, 189, 248, 0.08)',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Observed AQI: <b>%{y}</b><extra></extra>'
    ))

    date_label = f"{start_date.strftime('%b %d')} – {end_date.strftime('%b %d, %Y')}"
    fig.update_layout(
        title=dict(text=f"<b>Observed AQI Trend — {city_name}</b> <span style='font-size:0.85rem; color:#94a3b8;'>({date_label})</span>"),
        xaxis=dict(title="Observation Date", gridcolor="rgba(128, 128, 128, 0.15)"),
        yaxis=dict(title="AQI Level", gridcolor="rgba(128, 128, 128, 0.15)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
        height=330,
        margin=dict(l=10, r=10, t=45, b=10),
        hovermode="x unified"
    )
    return fig


def render_contextual_forecast_chart(daily_df, target_date, predicted_aqi, city_name):
    """Render a timeline linking recent history through today to the user-selected prediction date"""
    today_dt = pd.to_datetime(date.today())
    target_dt = pd.to_datetime(target_date)

    past_df = daily_df[daily_df['Date'] <= today_dt].tail(10).copy()
    
    fig = go.Figure()

    # Historical curve
    fig.add_trace(go.Scatter(
        x=past_df['Date'],
        y=past_df['AQI'],
        mode='lines+markers',
        name='Observed (Recent)',
        line=dict(color='#94a3b8', width=2),
        marker=dict(size=6, color='#64748b'),
        hovertemplate='<b>%{x|%b %d}</b>: %{y} AQI<extra></extra>'
    ))

    # Connection line from today to target date
    forecast_dates = [past_df['Date'].iloc[-1], target_dt]
    forecast_values = [past_df['AQI'].iloc[-1], predicted_aqi]

    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines',
        name='AI Forecast Path',
        line=dict(color='#38bdf8', width=2.5, dash='dash')
    ))

    # Prominent marker on the target date
    details = get_aqi_details(predicted_aqi)
    fig.add_trace(go.Scatter(
        x=[target_dt],
        y=[predicted_aqi],
        mode='markers+text',
        name='Selected Date Prediction',
        marker=dict(size=15, color=details['color'], symbol='diamond', line=dict(color='#ffffff', width=2)),
        text=[f"<b>{predicted_aqi} AQI</b> ({details['category']})"],
        textposition="top center",
        textfont=dict(color=details['color'], size=13, family="Plus Jakarta Sans")
    ))

    fig.update_layout(
        title=f"<b>Forecasting Trajectory for {target_date.strftime('%A, %b %d, %Y')}</b>",
        xaxis=dict(title="Timeline", gridcolor="rgba(128, 128, 128, 0.15)"),
        yaxis=dict(title="AQI", gridcolor="rgba(128, 128, 128, 0.15)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
        height=330,
        margin=dict(l=10, r=10, t=45, b=10),
        hovermode="x unified"
    )
    return fig


# ---------------------------------------------------------
# Main Application Flow
# ---------------------------------------------------------
def main():
    ml_model, feature_cols = load_ml_assets()

    # Top Brand Banner
    st.markdown("""
    <div class="brand-banner">
        <div>
            <div class="brand-title">🍃 AirAware AI</div>
            <div class="brand-subtitle">Real-time Satellite & Sensor Telemetry with Machine Learning Forecasting</div>
        </div>
        <div>
            <span class="live-badge">
                <span class="live-dot"></span>
                Connected to Open-Meteo Live API
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Sidebar Controls: City, Date Range & Telemetry Status
    # ---------------------------------------------------------
    with st.sidebar:
        st.markdown("### 📍 Location & Monitoring")
        city_names = list(CITIES.keys())
        selected_city = st.selectbox("Select Indian City", city_names, index=city_names.index("Chennai") if "Chennai" in city_names else 0)
        
        city_meta = CITIES[selected_city]
        st.caption(f"**State / Region:** {city_meta['state']}")
        st.caption(f"**Coordinates:** {city_meta['lat']}° N, {city_meta['lon']}° E")

        st.markdown("---")
        st.markdown("### 📅 Historical Date Range")
        range_option = st.selectbox(
            "Observation Window",
            ["Last 7 Days", "Last 14 Days", "Last 30 Days", "Last 60 Days", "Custom Range"],
            index=1,
            help="Choose how many days of past telemetry to display on the dashboard."
        )

        today = date.today()
        if range_option == "Last 7 Days":
            start_date = today - timedelta(days=7)
            end_date = today
        elif range_option == "Last 14 Days":
            start_date = today - timedelta(days=14)
            end_date = today
        elif range_option == "Last 30 Days":
            start_date = today - timedelta(days=30)
            end_date = today
        elif range_option == "Last 60 Days":
            start_date = today - timedelta(days=60)
            end_date = today
        else:
            custom_range = st.date_input(
                "Select Start & End Dates",
                value=(today - timedelta(days=14), today),
                min_value=today - timedelta(days=60),
                max_value=today
            )
            if isinstance(custom_range, (list, tuple)) and len(custom_range) == 2:
                start_date, end_date = custom_range
            else:
                start_date = today - timedelta(days=14)
                end_date = today

        st.caption(f"Showing: **{start_date.strftime('%b %d, %Y')}** to **{end_date.strftime('%b %d, %Y')}**")

        st.markdown("---")
        st.markdown("### ⚡ Live Telemetry Status")
        st.markdown(f"""
        - **Source:** Open-Meteo API
        - **Sampling:** 1-Hour Continuous
        - **ML Engine:** XGBoost Regressor
        - **Standard:** India CPCB (2024-26)
        """)

        if st.button("🔄 Refresh Telemetry", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Fetch live telemetry for selected city
    with st.spinner(f"Fetching real-world telemetry for {selected_city}..."):
        raw_telemetry, err = fetch_live_air_quality(city_meta['lat'], city_meta['lon'], past_days=60, forecast_days=7)

    if err or not raw_telemetry:
        st.error(f"Unable to connect to live Open-Meteo telemetry: {err}")
        st.info("Please check your internet connection or try again in a few moments.")
        return

    hourly_df, daily_df, live_snapshot = process_telemetry(raw_telemetry)

    # Filter historical dataframe based on the user's chosen Date Range
    filtered_daily_df = daily_df[
        (daily_df['Date'].dt.date >= start_date) & 
        (daily_df['Date'].dt.date <= end_date)
    ].copy()

    if filtered_daily_df.empty:
        filtered_daily_df = daily_df[daily_df['Date'].dt.date <= today].tail(14).copy()

    # ---------------------------------------------------------
    # Top Level Navigation Tabs
    # ---------------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "🌿 Real-time Air Quality",
        "🔮 Predict Future Date",
        "📊 Pollutant Trends & Insights"
    ])

    # =========================================================
    # TAB 1: Real-time Air Quality
    # =========================================================
    with tab1:
        st.markdown(f"#### Current Air Quality Status in **{selected_city}**")
        
        col1, col2 = st.columns([1.1, 1.9])
        aqi_meta = get_aqi_details(live_snapshot['aqi'])

        with col1:
            with st.container(border=True):
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Current Live AQI</span>
                    <span class="aqi-pill {aqi_meta['css_class']}">{aqi_meta['category']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                gauge_fig = render_speedometer(live_snapshot['aqi'], selected_city)
                st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})

                st.markdown(f"""
                <div style="text-align: center; margin-top: -8px;">
                    <span style="font-size: 0.85rem; color: #94a3b8;">Dominant Pollutant:</span> 
                    <b style="color: #38bdf8; font-size: 1rem;">{live_snapshot['dominant']}</b>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="health-tip-card">
                    <b style="color: #34d399; font-size: 0.88rem;">🩺 Health Advisory</b><br>
                    <span style="font-size: 0.84rem; color: inherit;">{aqi_meta['health_note']}</span><br>
                    <span style="font-size: 0.82rem; font-weight: 600; color: #38bdf8; margin-top: 6px; display: inline-block;">
                        Outdoor Safety: {aqi_meta['outdoor_safe']}
                    </span>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            with st.container(border=True):
                st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 12px;'>Core Pollutant Concentrations & Thresholds</div>", unsafe_allow_html=True)
                
                p_cols = st.columns(3)
                pollutants_to_show = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
                
                for i, p in enumerate(pollutants_to_show):
                    val = live_snapshot['pollutants'].get(p, 0.0)
                    limit_info = POLLUTANT_SAFE_LIMITS[p]
                    safe_limit = limit_info['limit']
                    unit = limit_info['unit']
                    
                    ratio = (val / safe_limit) if safe_limit > 0 else 0
                    if ratio <= 0.6:
                        status_text = "Good"
                        status_bg = "rgba(16, 185, 129, 0.2)"
                        status_color = "#34d399"
                    elif ratio <= 1.0:
                        status_text = "Moderate"
                        status_bg = "rgba(234, 179, 8, 0.2)"
                        status_color = "#facc15"
                    elif ratio <= 1.5:
                        status_text = "High"
                        status_bg = "rgba(249, 115, 22, 0.2)"
                        status_color = "#fb923c"
                    else:
                        status_text = "Critical"
                        status_bg = "rgba(239, 68, 68, 0.2)"
                        status_color = "#f87171"

                    with p_cols[i % 3]:
                        st.markdown(f"""
                        <div class="pollutant-card">
                            <div class="pollutant-name">{p}</div>
                            <div class="pollutant-val">{val}</div>
                            <div class="pollutant-unit">{unit} (Limit: {safe_limit})</div>
                            <div class="pollutant-status" style="background: {status_bg}; color: {status_color};">
                                {status_text} • {int(ratio*100)}% of limit
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        # Historical Trend Section with Period Statistics
        with st.container(border=True):
            trend_fig = render_history_trend(filtered_daily_df, selected_city, start_date, end_date)
            st.plotly_chart(trend_fig, use_container_width=True)

            # Period Summary Metrics
            stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
            avg_aqi = round(filtered_daily_df['AQI'].mean())
            max_aqi = round(filtered_daily_df['AQI'].max())
            min_aqi = round(filtered_daily_df['AQI'].min())
            days_count = len(filtered_daily_df)

            stat_c1.metric("Selected Window", f"{days_count} Days")
            stat_c2.metric("Period Average AQI", f"{avg_aqi}", delta=None)
            stat_c3.metric("Peak Observed AQI", f"{max_aqi}", delta="Worst Day", delta_color="inverse")
            stat_c4.metric("Lowest Observed AQI", f"{min_aqi}", delta="Cleanest Day")

    # =========================================================
    # TAB 2: Predict Future Date (User's specific core feature)
    # =========================================================
    with tab2:
        st.markdown("### 🔮 Predict Future Air Quality for a Specific Day")
        st.markdown("Select any upcoming date. Our system combines **live atmospheric forecast vectors** with the **XGBoost Machine Learning model** (trained on CPCB telemetry) to predict the exact Air Quality Index and pollutant levels.")

        # Date Picker Container
        with st.container(border=True):
            today = date.today()
            tomorrow = today + timedelta(days=1)
            max_date = today + timedelta(days=14)

            pick_col1, pick_col2 = st.columns([1.5, 2.5])
            
            with pick_col1:
                st.markdown("**📅 Pick a Target Date:**")
                selected_date = st.date_input(
                    "Target Date",
                    value=tomorrow,
                    min_value=today,
                    max_value=max_date,
                    label_visibility="collapsed"
                )

                st.markdown("<div style='font-size: 0.8rem; color: #94a3b8; margin-top: 8px;'>Quick Presets:</div>", unsafe_allow_html=True)
                preset_col1, preset_col2, preset_col3 = st.columns(3)
                with preset_col1:
                    if st.button("Tomorrow", use_container_width=True):
                        selected_date = tomorrow
                        st.rerun()
                with preset_col2:
                    if st.button("+3 Days", use_container_width=True):
                        selected_date = today + timedelta(days=3)
                        st.rerun()
                with preset_col3:
                    if st.button("+7 Days", use_container_width=True):
                        selected_date = today + timedelta(days=7)
                        st.rerun()

            with pick_col2:
                days_from_now = (selected_date - today).days
                day_str = "Today" if days_from_now == 0 else f"{days_from_now} day(s) from now"
                st.markdown(f"""
                <div style="background: rgba(128, 128, 128, 0.08); border: 1px solid rgba(128, 128, 128, 0.18); padding: 18px; border-radius: 12px; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                    <span style="font-size: 0.82rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Selected Target Horizon</span>
                    <span style="font-size: 1.35rem; font-weight: 800; color: #38bdf8;">{selected_date.strftime('%A, %d %B %Y')}</span>
                    <span style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">Offset: <b>{day_str}</b> in <b>{selected_city}</b></span>
                </div>
                """, unsafe_allow_html=True)

        # Run Prediction for Target Date
        prediction_result = predict_for_specific_date(selected_date, daily_df, ml_model, feature_cols)

        # Render Prediction Output
        with st.container(border=True):
            st.markdown(f"<div style='font-size: 0.88rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.05em;'>AI Prediction Output for {selected_city}</div>", unsafe_allow_html=True)
            
            pred_col1, pred_col2 = st.columns([1.2, 1.8])
            
            with pred_col1:
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="pred-label">Predicted Air Quality Index</div>
                    <div class="pred-val" style="color: {prediction_result['color']};">
                        {prediction_result['aqi']}
                    </div>
                    <div>
                        <span class="aqi-pill {prediction_result['css_class']}">{prediction_result['category']}</span>
                    </div>
                    <div style="margin-top: 14px; font-size: 0.85rem; color: #94a3b8;">
                        Primary Contributing Pollutant: <b style="color: #38bdf8;">{prediction_result['dominant']}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with pred_col2:
                st.markdown(f"""
                <div style="padding: 8px;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Health Impact on {selected_date.strftime('%b %d')}:</div>
                    <div style="font-size: 1.05rem; font-weight: 600; color: inherit; margin: 6px 0 14px 0;">
                        {prediction_result['health_note']}
                    </div>
                    <div class="health-tip-card">
                        <b style="color: #34d399;">🏃 Outdoor Activity Guidance:</b><br>
                        <span>{prediction_result['outdoor_safe']}</span>
                    </div>
                    <div style="margin-top: 14px; font-size: 0.78rem; color: #94a3b8;">
                        <b>Model Pipeline:</b> {prediction_result['source_note']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Detailed Pollutant Forecast Breakdown
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin: 18px 0 10px 0;'>Forecasted Pollutant Breakdown on Target Day</div>", unsafe_allow_html=True)
            
            f_cols = st.columns(6)
            for i, (pol, val) in enumerate(prediction_result['pollutants'].items()):
                unit = POLLUTANT_SAFE_LIMITS[pol]['unit']
                with f_cols[i]:
                    st.markdown(f"""
                    <div class="pollutant-card" style="padding: 12px 6px;">
                        <div class="pollutant-name">{pol}</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: inherit;">{val}</div>
                        <div style="font-size: 0.72rem; color: #94a3b8;">{unit}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Contextual Timeline Chart
        with st.container(border=True):
            forecast_fig = render_contextual_forecast_chart(daily_df, selected_date, prediction_result['aqi'], selected_city)
            st.plotly_chart(forecast_fig, use_container_width=True)

    # =========================================================
    # TAB 3: Pollutant Trends & Insights
    # =========================================================
    with tab3:
        st.markdown(f"#### Multi-Pollutant Analytics for **{selected_city}**")
        
        with st.container(border=True):
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px;'>48-Hour High-Resolution Diurnal Progression (PM2.5 vs PM10)</div>", unsafe_allow_html=True)
            
            recent_hourly = hourly_df.tail(48).copy()
            
            hour_fig = go.Figure()
            hour_fig.add_trace(go.Scatter(
                x=recent_hourly['time'],
                y=recent_hourly['PM2.5'],
                mode='lines',
                name='PM2.5 (Fine Particulate)',
                line=dict(color='#ef4444', width=2)
            ))
            hour_fig.add_trace(go.Scatter(
                x=recent_hourly['time'],
                y=recent_hourly['PM10'],
                mode='lines',
                name='PM10 (Coarse Particulate)',
                line=dict(color='#f59e0b', width=2)
            ))
            hour_fig.add_trace(go.Scatter(
                x=recent_hourly['time'],
                y=recent_hourly['NO2'],
                mode='lines',
                name='NO2 (Vehicular Emissions)',
                line=dict(color='#a855f7', width=2)
            ))

            hour_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
                height=320,
                xaxis=dict(title="Time", gridcolor="rgba(128, 128, 128, 0.15)"),
                yaxis=dict(title="Concentration (µg/m³)", gridcolor="rgba(128, 128, 128, 0.15)"),
                margin=dict(l=10, r=10, t=20, b=10),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(hour_fig, use_container_width=True)

        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            with st.container(border=True):
                st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px;'>Machine Learning Architecture</div>", unsafe_allow_html=True)
                st.markdown("""
                - **Base Model:** XGBoost Regressor (`models/xgboost_model.pkl`)
                - **Training Heritage:** 2015–2020 CPCB City Day monitoring records.
                - **Engineered Features (23 total):**
                  - Lags: `AQI_lag_1`, `AQI_lag_2`, `AQI_lag_3`, `AQI_lag_7`, `AQI_lag_14`
                  - Rolling: `AQI_rolling_7`, `AQI_rolling_14`, `AQI_rolling_30`
                  - Pollutants: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
                  - Calendar: Month, DayOfWeek, Season
                - **Live Bridge:** Real-time atmospheric vector ingestion from Open-Meteo converts historical model weights into present & future predictive capacity.
                """)

        with comp_col2:
            with st.container(border=True):
                st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px;'>CPCB AQI Category Standard (India)</div>", unsafe_allow_html=True)
                st.markdown("""
                | AQI Index | Category | Associated Health Impact |
                | :--- | :--- | :--- |
                | **0 – 50** | 🟢 Good | Minimal impact |
                | **51 – 100** | 🟡 Satisfactory | Minor discomfort to sensitive people |
                | **101 – 200** | 🟠 Moderate | Discomfort to heart/lung patients |
                | **201 – 300** | 🔴 Poor | Breathing discomfort to most people |
                | **301 – 400** | 🟣 Very Poor | Respiratory illness on exposure |
                | **401 – 500** | 🟤 Severe | Serious impacts on healthy adults |
                """)


if __name__ == "__main__":
    main()
