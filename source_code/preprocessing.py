"""
Data preprocessing pipeline:
- Load raw sales data
- Handle missing values (zero-fill / mean substitution)
- Aggregate to daily granularity
- Min-max normalisation
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_sales(path=None):
    path = path or os.path.join(DATA_DIR, "sales_data.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def aggregate_daily(df, store_id=None, sku_id=None):
    """Aggregate to a single (store, sku) daily time series."""
    sub = df.copy()
    if store_id is not None:
        sub = sub[sub["store_id"] == store_id]
    if sku_id is not None:
        sub = sub[sub["sku_id"] == sku_id]
    daily = (sub.groupby("date", as_index=False)["units_sold"]
                .sum()
                .sort_values("date"))
    return daily


def handle_missing(df, value_col="units_sold"):
    df = df.copy()
    df[value_col] = df[value_col].fillna(0)
    df.loc[df[value_col] < 0, value_col] = 0
    return df


def min_max_scale(series):
    s_min = float(series.min())
    s_max = float(series.max())
    if s_max - s_min == 0:
        return series * 0.0, (s_min, s_max)
    scaled = (series - s_min) / (s_max - s_min)
    return scaled, (s_min, s_max)


def inverse_scale(scaled, scaler):
    s_min, s_max = scaler
    return scaled * (s_max - s_min) + s_min


def train_val_test_split(series, train_frac=0.70, val_frac=0.15):
    n = len(series)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return series[:train_end], series[train_end:val_end], series[val_end:]
