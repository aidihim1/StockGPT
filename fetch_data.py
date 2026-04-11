# fetch_data.py  —  fetches historical daily prices for all NSE stocks

import pandas as pd
import time
import os
from datetime import datetime, timedelta
from login import get_api

# ── Settings ──────────────────────────────────────────────────────────────────
YEARS_OF_DATA  = 5          # how many years back to fetch (Angel allows ~5 years max)
SLEEP_BETWEEN  = 0.4        # seconds to wait between API calls (avoid rate limits)
OUTPUT_FILE    = "returns_dataset.csv"
RAW_FOLDER     = "raw_prices"   # folder to save individual stock CSVs
# ──────────────────────────────────────────────────────────────────────────────

def fetch_one_stock(api, token, symbol, from_date, to_date):
    """Fetch daily OHLC data for one stock. Returns a DataFrame or None."""
    try:
        params = {
            "exchange":    "NSE",
            "symboltoken": token,
            "interval":    "ONE_DAY",      # daily candles
            "fromdate":    from_date,      # format: "YYYY-MM-DD HH:MM"
            "todate":      to_date,
        }
        response = api.getCandleData(params)
        
        if response["status"] and response["data"]:
            df = pd.DataFrame(
                response["data"],
                columns=["datetime", "open", "high", "low", "close", "volume"]
            )
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.date
            df["symbol"]   = symbol
            return df
        else:
            return None
            
    except Exception as e:
        print(f"  ERROR fetching {symbol}: {e}")
        return None


def build_dataset():
    """Main function: fetch all stocks and build the returns dataset."""
    
    # 1. Login
    print("=== Logging in to Angel One ===")
    api = get_api()
    
    # 2. Load symbol list
    if not os.path.exists("nse_symbols.csv"):
        print("ERROR: nse_symbols.csv not found. Run fetch_symbols.py first!")
        return
    
    symbols_df = pd.read_csv("nse_symbols.csv")
    print(f"Found {len(symbols_df)} NSE stocks to fetch.\n")
    
    # 3. Set date range
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=365 * YEARS_OF_DATA)
    from_date  = start_date.strftime("%Y-%m-%d %H:%M")
    to_date    = end_date.strftime("%Y-%m-%d %H:%M")
    print(f"Fetching data from {from_date} to {to_date}\n")
    
    # 4. Create folder for raw files
    os.makedirs(RAW_FOLDER, exist_ok=True)
    
    # 5. Loop through every stock
    all_closes = {}   # dict: symbol → Series of closing prices
    
    for i, row in symbols_df.iterrows():
        token  = str(row["token"])
        symbol = row["clean_symbol"]
        
        # Skip if already downloaded
        raw_path = f"{RAW_FOLDER}/{symbol}.csv"
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path, parse_dates=["datetime"])
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.date
        else:
            df = fetch_one_stock(api, token, symbol, from_date, to_date)
            if df is not None:
                df.to_csv(raw_path, index=False)
            time.sleep(SLEEP_BETWEEN)   # be polite to the API
        
        if df is not None and len(df) > 100:   # skip stocks with too little data
            closes = df.set_index("datetime")["close"]
            all_closes[symbol] = closes
        
        # Progress update every 50 stocks
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(symbols_df)} stocks fetched...")
    
    print(f"\nSuccessfully fetched {len(all_closes)} stocks with enough data.")
    
    # 6. Build a wide price matrix (rows = dates, columns = stocks)
    print("\nBuilding price matrix...")
    price_matrix = pd.DataFrame(all_closes)
    price_matrix.sort_index(inplace=True)
    
    # 7. Compute daily returns: (today / yesterday) - 1
    print("Computing daily returns...")
    returns_matrix = price_matrix.pct_change()  # pct_change = (new-old)/old
    
    # Drop the first row (it's NaN since there's no "previous day")
    returns_matrix = returns_matrix.iloc[1:]
    
    # 8. Save to CSV
    returns_matrix.to_csv(OUTPUT_FILE)
    print(f"\nDone! Dataset saved to '{OUTPUT_FILE}'")
    print(f"Shape: {returns_matrix.shape[0]} trading days × {returns_matrix.shape[1]} stocks")
    print(f"\nSample (first 5 rows, first 5 stocks):")
    print(returns_matrix.iloc[:5, :5].round(4))


if __name__ == "__main__":
    build_dataset()