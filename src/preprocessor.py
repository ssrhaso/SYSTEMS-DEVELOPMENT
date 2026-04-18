"""
Preprocessing Script 

Ishaq Modassir Mushtaq - 24030388 
"""

import os
import pandas as pd

_BASE = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
COFFEE_PATH = os.path.join(_BASE, "Pink_CoffeeSales_March - Oct 2025.csv")
CROISSANT_PATH = os.path.join(_BASE, "Pink_CroissantSales_March-Oct_2025.csv")


def _read_coffee(file_path: str) -> pd.DataFrame:
    header_row1 = pd.read_csv(file_path, nrows=0).columns.tolist()
    header_row2 = pd.read_csv(file_path, skiprows=1, nrows=0).columns.tolist()

    combined_headers = []
    for i, (col1, col2) in enumerate(zip(header_row1, header_row2)):
        if i == 0 and col1 and not col1.startswith("Unnamed"):
            combined_headers.append(col1)
        elif col2 and not col2.startswith("Unnamed") and col2.strip():
            combined_headers.append(col2)
        elif col1 and not col1.startswith("Unnamed"):
            combined_headers.append(col1)
        else:
            combined_headers.append(f"Column_{i}")

    data = pd.read_csv(file_path, skiprows=1)
    data.columns = combined_headers
    return data


def _read_croissant(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    if "Number Sold" in data.columns:
        data = data.rename(columns={"Number Sold": "Croissants"})
    return data


def _clean(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y", errors="coerce")
    data = data.dropna()
    data = data.drop_duplicates()
    data = data.sort_values("Date").reset_index(drop=True)
    return data


def load_coffee() -> pd.DataFrame:
    return _clean(_read_coffee(COFFEE_PATH))


def load_croissant() -> pd.DataFrame:
    return _clean(_read_croissant(CROISSANT_PATH))


def load_all() -> pd.DataFrame:
    coffee = load_coffee()
    croissant = load_croissant()
    return pd.merge(coffee, croissant, on="Date", how="inner")


def to_series(df: pd.DataFrame, product: str) -> pd.DataFrame:
    if product not in df.columns:
        raise ValueError(
            f"Product '{product}' not found. "
            f"Available: {[c for c in df.columns if c != 'Date']}"
        )
    series = df[["Date", product]].rename(columns={"Date": "ds", product: "y"})
    series["y"] = pd.to_numeric(series["y"], errors="coerce")
    return series.dropna().sort_values("ds").reset_index(drop=True)


if __name__ == "__main__":
    coffee = load_coffee()
    print(coffee.head())
    print(f"Shape: {coffee.shape}, Columns: {coffee.columns.tolist()}")

    croissant = load_croissant()
    print(croissant.head())
    print(f"Shape: {croissant.shape}, Columns: {croissant.columns.tolist()}")

    all_data = load_all()
    print(all_data.head())
    print(f"Shape: {all_data.shape}, Columns: {all_data.columns.tolist()}")

    series = to_series(all_data, "Cappuccino")
    print(series.head())
