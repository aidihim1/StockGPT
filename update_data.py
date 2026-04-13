# update_data.py -- fetches latest prices from Angel One and updates dataset.csv
# Falls back to yfinance if Angel One API is unavailable (e.g. after market hours)
# Run this EVERY DAY before running forecast.py to keep data current

import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from login import get_api

SLEEP_BETWEEN = 0.1          # reduced from 0.4 -> ~4 min for 2411 stocks
SAVE_EVERY    = 300          # save progress to CSV every N stocks
DATA_FILE     = "dataset.csv"
RAW_FOLDER    = "raw_prices"

def update_data():
    # 1. Load existing dataset to find last date
    print("Loading existing data ...")
    df = pd.read_csv(DATA_FILE, parse_dates=["date"],
                     usecols=["date", "stock", "open", "high", "low", "close", "volume", "return_1d"])
    last_date = df["date"].max()
    today     = datetime.now().date()

    print(f"  Last date in dataset : {last_date.date()}")
    print(f"  Today                : {today}")

    if last_date.date() >= today:
        print("Data is already up to date. Nothing to fetch.")
        return

    days_behind = (today - last_date.date()).days
    print(f"  Days behind          : {days_behind} calendar days\n")

    # 2. Login (optional -- fall back to yfinance if Angel One is unavailable)
    print("Logging in to Angel One ...")
    api = None
    try:
        api = get_api()
        print("  Angel One login successful.")
    except Exception as e:
        print(f"  Angel One login failed ({type(e).__name__}). Using yfinance fallback.")

    # 3. Load symbols
    symbols_df = pd.read_csv("nse_symbols.csv")
    existing_stocks = set(df["stock"].unique())

    # Fetch from 7 days before last_date to ensure return computation is correct
    from_date = (last_date - timedelta(days=7)).strftime("%Y-%m-%d %H:%M")
    to_date   = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"Fetching new data from {from_date} to {to_date} ...\n")

    new_rows  = []
    saved_count = 0

    for i, row in symbols_df.iterrows():
        token  = str(row["token"])
        symbol = row["clean_symbol"]

        if symbol not in existing_stocks:
            continue

        fetched_ok = False
        if api is not None:
            try:
                params = {
                    "exchange":    "NSE",
                    "symboltoken": token,
                    "interval":    "ONE_DAY",
                    "fromdate":    from_date,
                    "todate":      to_date,
                }
                resp = api.getCandleData(params)
                if resp["status"] and resp["data"]:
                    tmp = pd.DataFrame(resp["data"],
                                       columns=["datetime", "open", "high", "low", "close", "volume"])
                    tmp["date"] = pd.to_datetime(tmp["datetime"]).dt.normalize()
                    tmp = tmp.sort_values("date")
                    tmp["return_1d"] = tmp["close"].pct_change().clip(-1.0, 1.0)
                    tmp["stock"] = symbol
                    tmp = tmp[tmp["date"] > pd.Timestamp(last_date)]
                    tmp = tmp.dropna(subset=["return_1d"])
                    if len(tmp) > 0:
                        tmp = tmp[["date", "stock", "open", "high", "low", "close", "volume", "return_1d"]]
                        new_rows.append(tmp)
                    fetched_ok = True
            except Exception:
                pass

        if not fetched_ok:
            # Fall back to yfinance
            try:
                tmp = _fetch_yfinance(symbol, last_date)
                if tmp is not None:
                    new_rows.append(tmp)
            except Exception:
                pass

        time.sleep(SLEEP_BETWEEN)

        fetched = i + 1
        if fetched % 100 == 0:
            print(f"  {fetched}/{len(symbols_df)} stocks fetched ...")

        # Incremental save every SAVE_EVERY stocks -- so partial runs keep progress
        if new_rows and fetched % SAVE_EVERY == 0:
            _partial = pd.concat(new_rows, axis=0, ignore_index=True)
            _partial = _round_df(_partial)
            _combined = pd.concat([df, _partial], axis=0, ignore_index=True)
            _combined = _combined.drop_duplicates(subset=["date", "stock"], keep="last")
            _combined = _combined.sort_values(["date", "stock"]).reset_index(drop=True)
            _combined.to_csv(DATA_FILE, index=False)
            saved_count = len(_partial)
            print(f"  [saved {saved_count} new rows so far -> {DATA_FILE}]")

    print(f"  {len(symbols_df)}/{len(symbols_df)} stocks processed.")

    if not new_rows:
        print("No new data received (market may have been closed).")
        return

    # 4. Build final new rows dataframe
    new_df = pd.concat(new_rows, axis=0, ignore_index=True)
    new_df  = _round_df(new_df)

    new_dates = sorted(new_df["date"].dt.date.unique())
    print(f"\n  New trading days found: {len(new_dates)}")
    for d in new_dates:
        print(f"    {d}")

    # 5. Final save to dataset.csv
    combined = pd.concat([df, new_df], axis=0, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "stock"], keep="last")
    combined = combined.sort_values(["date", "stock"]).reset_index(drop=True)
    combined.to_csv(DATA_FILE, index=False)

    print(f"\nUpdated {DATA_FILE}")
    print(f"  Total rows    : {len(combined):,}")
    print(f"  Stocks        : {combined['stock'].nunique()}")
    print(f"  Date range    : {combined['date'].min().date()} to {combined['date'].max().date()}")
    print("Done. Now run forecast.py for fresh predictions.")


def _fetch_yfinance(symbol, last_date):
    """Fetch missing days from yfinance for a single NSE symbol."""
    import yfinance as yf
    ticker = symbol + ".NS"
    from_dt = (last_date - timedelta(days=7)).date()
    try:
        raw = yf.download(ticker, start=str(from_dt), auto_adjust=True, progress=False)
        if raw is None or len(raw) == 0:
            return None
        raw = raw.reset_index()
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        raw = raw.rename(columns={"Date": "date", "Open": "open", "High": "high",
                                   "Low": "low", "Close": "close", "Volume": "volume"})
        raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
        raw = raw.sort_values("date")
        raw["return_1d"] = raw["close"].pct_change().clip(-1.0, 1.0)
        raw["stock"] = symbol
        raw = raw[raw["date"] > pd.Timestamp(last_date)]
        raw = raw.dropna(subset=["return_1d"])
        if len(raw) == 0:
            return None
        return raw[["date", "stock", "open", "high", "low", "close", "volume", "return_1d"]]
    except Exception:
        return None


def _round_df(new_df):
    new_df["open"]      = new_df["open"].round(4)
    new_df["high"]      = new_df["high"].round(4)
    new_df["low"]       = new_df["low"].round(4)
    new_df["close"]     = new_df["close"].round(4)
    new_df["volume"]    = new_df["volume"].fillna(0).astype(np.int64)
    new_df["return_1d"] = new_df["return_1d"].round(6)
    return new_df


if __name__ == "__main__":
    update_data()
