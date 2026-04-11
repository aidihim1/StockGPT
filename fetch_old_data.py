# fetch_old_data.py — pulls historical data via yfinance (free, no API key needed)

import yfinance as yf
import pandas as pd
import os
import time

OUTPUT_FOLDER = "raw_prices_old"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Read your existing symbol list
symbols_df = pd.read_csv("nse_symbols.csv")

# yfinance needs ".NS" suffix for NSE stocks (e.g. "RELIANCE.NS")
START_DATE = "2000-01-01"
END_DATE   = "2020-01-01"

all_closes = {}
failed     = []

for i, row in symbols_df.iterrows():
    symbol    = row["clean_symbol"]
    yf_symbol = symbol + ".NS"
    path      = f"{OUTPUT_FOLDER}/{symbol}.csv"

    # Skip if already downloaded
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Close" in df.columns:
            # Convert to numeric — drops ticker-name rows saved by old yfinance
            close = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if len(close) > 200:
                all_closes[symbol] = close
        continue

    try:
        df = yf.download(yf_symbol, start=START_DATE, end=END_DATE,
                         progress=False, auto_adjust=True)

        if df.empty or len(df) <= 200:
            print(f"  [{i+1}] {symbol}: skipped (only {len(df)} days)")
            failed.append(symbol)
            continue

        # Flatten MultiIndex columns if present (newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.to_csv(path)
        all_closes[symbol] = pd.to_numeric(df["Close"], errors="coerce").dropna()
        print(f"  [{i+1}] {symbol}: {len(df)} days")

    except Exception as e:
        print(f"  [{i+1}] {symbol}: ERROR — {e}")
        failed.append(symbol)

    time.sleep(0.3)   # be gentle with yfinance rate limits

# Build returns matrix for old data
print(f"\nBuilding old returns matrix from {len(all_closes)} stocks...")
if not all_closes:
    print("No data collected — nothing to save.")
else:
    price_matrix   = pd.DataFrame(all_closes).sort_index()
    returns_matrix = price_matrix.pct_change().iloc[1:]
    returns_matrix.to_csv("returns_old.csv")
    print(f"Saved returns_old.csv — shape: {returns_matrix.shape}")

print(f"Failed/skipped: {len(failed)} stocks")