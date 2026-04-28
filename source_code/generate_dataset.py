"""
Synthetic dataset generator for the AI demand forecasting research paper.
Produces realistic kirana store sales data with weekly cycles, monthly
seasonality, and festival-driven demand spikes.

Usage:
    python generate_dataset.py
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

START_DATE = date(2022, 4, 1)
END_DATE = date(2023, 9, 30)
NUM_STORES = 5
NUM_SKUS = 312

CATEGORIES = [
    "groceries", "dairy", "beverages", "personal_care",
    "household", "snacks", "staples"
]

FESTIVALS = {
    "diwali_2022":   [date(2022, 10, 22), date(2022, 10, 26)],
    "diwali_2023":   [date(2023, 11, 12), date(2023, 11, 14)],
    "holi_2023":     [date(2023, 3, 7),   date(2023, 3, 8)],
    "eid_2023":      [date(2023, 4, 22),  date(2023, 4, 22)],
    "christmas_2022":[date(2022, 12, 24), date(2022, 12, 25)],
}

CATEGORY_FESTIVAL_MULTIPLIER = {
    "snacks":         4.0,
    "beverages":      3.2,
    "household":      2.5,
    "groceries":      2.5,
    "staples":        2.0,
    "personal_care":  1.8,
    "dairy":          1.5,
}

WEEKLY_PATTERN = {
    "snacks":        [1.00, 0.95, 0.95, 1.00, 1.10, 1.35, 1.20],
    "beverages":     [1.00, 0.95, 0.95, 1.00, 1.15, 1.35, 1.20],
    "dairy":         [1.05, 1.00, 1.00, 1.00, 1.05, 1.20, 1.15],
    "groceries":     [1.00, 0.98, 0.98, 1.00, 1.05, 1.25, 1.10],
    "household":     [1.00, 0.95, 0.95, 0.98, 1.05, 1.25, 1.10],
    "personal_care": [1.00, 0.95, 0.98, 1.00, 1.05, 1.20, 1.05],
    "staples":       [1.00, 0.98, 0.98, 1.02, 1.05, 1.18, 1.10],
}

CATEGORY_BASE_DEMAND = {
    "groceries":     30,
    "dairy":         25,
    "beverages":     18,
    "personal_care": 8,
    "household":     6,
    "snacks":        20,
    "staples":       22,
}


def in_festival(d):
    for name, (start, end) in FESTIVALS.items():
        if start <= d <= end:
            return name
    return None


def get_festival_multiplier(category, festival_name):
    if festival_name is None:
        return 1.0
    base = CATEGORY_FESTIVAL_MULTIPLIER.get(category, 2.0)
    if "diwali" in festival_name:
        return base
    if "holi" in festival_name:
        return base * 0.75
    return 1.5


def get_monthly_multiplier(category, month):
    if month in (10, 11):
        return 1.20
    if month == 3:
        return 1.10
    if month in (5, 6):
        return 1.10 if category == "beverages" else 0.95
    if month == 12:
        return 1.10
    return 1.0


def build_skus():
    rows = []
    np.random.seed(42)
    for i in range(NUM_SKUS):
        category = CATEGORIES[i % len(CATEGORIES)]
        sku_id = f"SKU{i+1:04d}"
        base = CATEGORY_BASE_DEMAND[category]
        per_sku_base = max(2, int(np.random.normal(base, base * 0.35)))
        rows.append({
            "sku_id": sku_id,
            "category": category,
            "base_demand": per_sku_base,
        })
    return pd.DataFrame(rows)


def generate_sales(skus_df):
    rng = np.random.default_rng(seed=7)
    days = (END_DATE - START_DATE).days + 1
    all_dates = [START_DATE + timedelta(days=i) for i in range(days)]

    records = []
    for store_id in range(1, NUM_STORES + 1):
        store_factor = rng.uniform(0.85, 1.20)

        for _, sku in skus_df.iterrows():
            sku_id = sku["sku_id"]
            category = sku["category"]
            base = sku["base_demand"]

            for d in all_dates:
                trend = 1.0 + 0.0002 * (d - START_DATE).days
                weekly = WEEKLY_PATTERN[category][d.weekday()]
                monthly = get_monthly_multiplier(category, d.month)
                fest_name = in_festival(d)
                fest = get_festival_multiplier(category, fest_name)
                noise = rng.normal(1.0, 0.18)

                demand = base * store_factor * trend * weekly * monthly * fest * noise
                demand = max(0, int(round(demand)))

                if demand == 0:
                    num_transactions = 0
                else:
                    basket = rng.uniform(14.0, 20.0)
                    num_transactions = max(1, int(round(demand / basket)))

                records.append({
                    "date": d.isoformat(),
                    "store_id": f"S{store_id}",
                    "sku_id": sku_id,
                    "category": category,
                    "units_sold": demand,
                    "num_transactions": num_transactions,
                    "is_festival": int(fest_name is not None),
                    "festival_name": fest_name or "",
                })

    return pd.DataFrame(records)


def build_festivals_csv():
    rows = []
    for name, (start, end) in FESTIVALS.items():
        rows.append({
            "festival": name,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        })
    return pd.DataFrame(rows)


def main():
    print("Generating SKU master ...")
    skus = build_skus()
    skus.to_csv(os.path.join(OUT_DIR, "sku_master.csv"), index=False)

    print(f"Generating daily sales for {NUM_STORES} stores x {NUM_SKUS} SKUs ...")
    sales = generate_sales(skus)
    sales.to_csv(os.path.join(OUT_DIR, "sales_data.csv"), index=False)

    festivals = build_festivals_csv()
    festivals.to_csv(os.path.join(OUT_DIR, "festivals.csv"), index=False)

    total_transactions = int(sales["num_transactions"].sum())
    print(f"Generated {len(sales):,} daily aggregated rows.")
    print(f"Total transactions (sum of num_transactions): {total_transactions:,}")
    print("Files written to:")
    print(f"  {OUT_DIR}/sku_master.csv")
    print(f"  {OUT_DIR}/sales_data.csv")
    print(f"  {OUT_DIR}/festivals.csv")


if __name__ == "__main__":
    main()
