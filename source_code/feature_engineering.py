"""
Feature engineering for demand forecasting.

Builds:
- Lag features (t-1, t-7, t-14)
- Rolling-window features (7-day rolling mean / std)
- Cyclic temporal encodings (sine-cosine for day-of-week, month)
- Calendar features (week-of-month)
- Festival binary flags
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LAGS = [1, 7, 14]


def add_lag_features(df, value_col="units_sold", lags=LAGS):
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[value_col].shift(lag)
    return df


def add_rolling_features(df, value_col="units_sold"):
    df = df.copy()
    df["roll_mean_7"] = df[value_col].shift(1).rolling(window=7).mean()
    df["roll_std_7"] = df[value_col].shift(1).rolling(window=7).std()
    return df


def add_temporal_features(df, date_col="date"):
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    dow = dates.dt.dayofweek
    month = dates.dt.month
    day = dates.dt.day

    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    df["week_of_month"] = ((day - 1) // 7) + 1
    df["is_weekend"] = (dow >= 5).astype(int)
    return df


def load_festival_calendar():
    path = os.path.join(DATA_DIR, "festivals.csv")
    fest = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    return fest


def add_festival_flags(df, date_col="date"):
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    fest = load_festival_calendar()
    flag = pd.Series(0, index=df.index)
    for _, row in fest.iterrows():
        mask = (dates >= row["start_date"]) & (dates <= row["end_date"])
        flag = flag | mask.astype(int).values
    df["is_festival"] = flag
    return df


def build_feature_frame(daily_df, value_col="units_sold"):
    df = daily_df.copy()
    df = add_lag_features(df, value_col=value_col)
    df = add_rolling_features(df, value_col=value_col)
    df = add_temporal_features(df, date_col="date")
    df = add_festival_flags(df, date_col="date")
    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14",
    "roll_mean_7", "roll_std_7",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "week_of_month", "is_weekend",
    "is_festival",
]
