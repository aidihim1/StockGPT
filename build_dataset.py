# build_dataset.py -- rebuilds the dataset in clean long format
#
# Output: dataset.csv with columns:
#   date, stock, open, high, low, close, volume, return_1d
#
# Each row = one stock on one date. No empty cells, no wide commas mess.

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

RAW_ANGEL  = "raw_prices"       # Angel One OHLCV CSVs (5yr)
RAW_YAHOO  = "raw_prices_old"   # Yahoo Finance OHLCV CSVs (pre-2020)
OUTPUT     = "dataset.csv"

def load_angel_stock(symbol: str) -> pd.DataFrame:
    """Load Angel One raw OHLCV and compute return."""
    path = f"{RAW_ANGEL}/{symbol}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    df["return_1d"] = df["close"].pct_change().clip(-1.0, 1.0)
    df["stock"] = symbol
    return df[["date", "stock", "open", "high", "low", "close", "volume", "return_1d"]]


def load_yahoo_stock(symbol: str) -> pd.DataFrame:
    """Load Yahoo Finance raw OHLCV and compute return."""
    path = f"{RAW_YAHOO}/{symbol}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Handle yfinance MultiIndex headers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Drop ticker-name row if present (old yfinance format)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(how="all")

    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")

    # Normalize column names (yfinance may vary)
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "open":   col_map[c] = "open"
        elif cl == "high":   col_map[c] = "high"
        elif cl == "low":    col_map[c] = "low"
        elif cl in ("close", "adj close"): col_map[c] = "close"
        elif cl == "volume": col_map[c] = "volume"
    df = df.rename(columns=col_map)

    needed = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return None

    df["return_1d"] = df["close"].pct_change().clip(-1.0, 1.0)
    df["stock"] = symbol
    df = df[needed + ["return_1d", "stock"]]
    df = df.rename(columns={"stock": "stock"})
    return df[["date", "stock", "open", "high", "low", "close", "volume", "return_1d"]]


def build_dataset():
    symbols_df = pd.read_csv("nse_symbols.csv")
    symbols    = symbols_df["clean_symbol"].tolist()
    total      = len(symbols)

    print(f"Building long-format dataset for {total} stocks ...")
    print(f"Sources: {RAW_ANGEL}/ (Angel One) + {RAW_YAHOO}/ (Yahoo pre-2020)\n")

    all_dfs = []
    skipped = 0

    for i, symbol in enumerate(symbols):
        frames = []

        # Yahoo (pre-2020 history)
        ydf = load_yahoo_stock(symbol)
        if ydf is not None and len(ydf) > 0:
            # Keep only pre-2020 rows from Yahoo to avoid overlap
            ydf = ydf[ydf["date"] < pd.Timestamp("2020-01-01")]
            if len(ydf) > 0:
                frames.append(ydf)

        # Angel One (5yr / recent)
        adf = load_angel_stock(symbol)
        if adf is not None and len(adf) > 0:
            frames.append(adf)

        if not frames:
            skipped += 1
            continue

        # Combine Yahoo + Angel for this stock, sort by date
        combined = pd.concat(frames, axis=0, ignore_index=True)
        combined = combined.sort_values("date").drop_duplicates("date")

        # Only keep rows where return is not NaN (first row always NaN due to pct_change)
        combined = combined.dropna(subset=["return_1d"])

        if len(combined) < 50:
            skipped += 1
            continue

        all_dfs.append(combined)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{total} processed ...")

    print(f"\nMerging {len(all_dfs)} stocks ...")
    dataset = pd.concat(all_dfs, axis=0, ignore_index=True)

    # Sort by date then stock for clean layout
    dataset = dataset.sort_values(["date", "stock"]).reset_index(drop=True)

    # Round floats for smaller file size
    for col in ["open", "high", "low", "close"]:
        dataset[col] = dataset[col].round(4)
    dataset["return_1d"] = dataset["return_1d"].round(6)
    dataset["volume"]    = dataset["volume"].fillna(0).astype(np.int64)

    # Save
    dataset.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")
    print(f"Shape : {dataset.shape[0]:,} rows x {dataset.shape[1]} columns")
    print(f"Stocks: {dataset['stock'].nunique()}")
    print(f"Dates : {dataset['date'].min().date()} to {dataset['date'].max().date()}")
    print(f"Skipped: {skipped} stocks (insufficient data)")
    print()
    print("=== Sample (first 10 rows) ===")
    print(dataset.head(10).to_string(index=False))


if __name__ == "__main__":
    build_dataset()
