# update_data.py -- fetches latest prices from Angel One and updates dataset.csv
# Falls back to yfinance for any stock Angel One fails on
# Run this before picks.py to keep data current (run_daily.bat does both)
#
# Each stock is fetched from ITS OWN last date (not the dataset-wide max), and the
# last REPAIR_DAYS are always re-fetched, so days missed by earlier failed runs get
# filled in. Stocks in nse_symbols.csv but not yet in dataset.csv are added.

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from clean_data import clean
from corporate_actions import candidate_stocks, fetch_actions, save_actions

SLEEP_BETWEEN  = 0.35         # Angel historical API allows ~3 req/sec
MAX_RETRIES    = 4            # retries per stock on rate-limit / network errors
ANGEL_GIVE_UP  = 10           # consecutive Angel failures before switching to yfinance-only
REPAIR_DAYS    = 120          # always re-fetch this many recent days (fills old gaps)
NEW_STOCK_DAYS = 1900         # history to fetch for stocks not yet in dataset (Angel max ~2000)
SAVE_EVERY     = 500          # save progress to CSV every N stocks
YF_BATCH       = 25           # tickers per yfinance batch (Yahoo rate-limits large requests)
YF_PAUSE       = 2            # seconds between yfinance batches
YF_PASSES      = 4            # retry passes for symbols Yahoo failed on
DATA_FILE      = "dataset.csv"
COLS           = ["date", "stock", "open", "high", "low", "close", "volume", "return_1d"]


def update_data():
    # 1. Load existing dataset and each stock's own last date
    print("Loading existing data ...")
    df = pd.read_csv(DATA_FILE, parse_dates=["date"], usecols=COLS)
    stock_last = df.groupby("stock")["date"].max()
    now   = datetime.now()
    today = pd.Timestamp(now.date())
    # Before market close today's candle is incomplete -- don't store it
    cutoff = today if now.hour * 60 + now.minute >= 15 * 60 + 45 else today - timedelta(days=1)

    print(f"  Last date in dataset : {df['date'].max().date()}")
    print(f"  Stocks behind latest : {(stock_last < stock_last.max()).sum()}")
    print(f"  Fetching up to       : {cutoff.date()}\n")

    # 2. Login (optional -- fall back to yfinance if Angel One is unavailable)
    print("Logging in to Angel One ...")
    api = None
    try:
        from login import get_api   # needs config.py (Angel One credentials)
        api = get_api()
        print("  Angel One login successful.")
    except Exception as e:
        print(f"  Angel One login failed ({type(e).__name__}: {e}). Using yfinance only.")

    # 3. Symbols (skip iNAV tickers -- indicative NAVs, not tradable, no candles)
    symbols_df = pd.read_csv("nse_symbols.csv")
    symbols_df = symbols_df[~symbols_df["clean_symbol"].str.endswith("INAV")].reset_index(drop=True)
    repair_from = today - timedelta(days=REPAIR_DAYS)

    def start_for(symbol):
        if symbol in stock_last.index:
            # 7-day overlap so the first new return is computed against a real prior close
            return min(stock_last[symbol], repair_from) - timedelta(days=7)
        return today - timedelta(days=NEW_STOCK_DAYS)

    to_date   = now.strftime("%Y-%m-%d %H:%M")
    new_rows  = []
    yf_queue  = []            # symbols Angel failed on
    n_angel   = 0
    angel_fail_streak = 0
    total     = len(symbols_df)
    print(f"Fetching {total} symbols ...\n")

    for i, row in symbols_df.iterrows():
        symbol = row["clean_symbol"]
        start  = start_for(symbol)

        tmp = None
        if api is not None:
            tmp = _fetch_angel(api, str(row["token"]), start, to_date)
            time.sleep(SLEEP_BETWEEN)

        if tmp is None:
            yf_queue.append(symbol)
            if api is not None:
                angel_fail_streak += 1
                if angel_fail_streak >= ANGEL_GIVE_UP:
                    print(f"  Angel One failed {ANGEL_GIVE_UP} stocks in a row -- "
                          f"switching to yfinance for the rest of this run.", flush=True)
                    api = None
        else:
            angel_fail_streak = 0
            n_angel += 1
            tmp = _finish(tmp, symbol, cutoff)
            if tmp is not None:
                new_rows.append(tmp)

        fetched = i + 1
        if fetched % 100 == 0:
            print(f"  {fetched}/{total} | angel ok {n_angel} | queued for yfinance {len(yf_queue)}", flush=True)

        # Incremental save -- so partial runs keep progress
        if new_rows and fetched % SAVE_EVERY == 0:
            _save(df, new_rows)
            print(f"  [progress saved -> {DATA_FILE}]", flush=True)

    # 4. yfinance fallback for everything Angel missed
    n_yf, failed = 0, []
    if yf_queue:
        print(f"\nFetching {len(yf_queue)} symbols from yfinance ...")
        yf_frames, failed = _fetch_yfinance_batch(yf_queue, {s: start_for(s) for s in yf_queue})
        for symbol, tmp in yf_frames.items():
            tmp = _finish(tmp, symbol, cutoff)
            if tmp is not None:
                new_rows.append(tmp)
                n_yf += 1

    print(f"\n  Angel One : {n_angel} stocks")
    print(f"  yfinance  : {n_yf} stocks")
    print(f"  Failed    : {len(failed)} stocks")
    if failed:
        print(f"    {', '.join(failed[:60])}{' ...' if len(failed) > 60 else ''}")

    if not new_rows:
        print("No new data received.")
        return

    # Record splits/bonuses for stocks with big moves in the new data (clean_data only adjusts
    # a move when a split/bonus is recorded -- it never guesses from the size of the drop)
    new_df = pd.concat(new_rows, ignore_index=True)
    cands = candidate_stocks(pd.concat([df[df["stock"].isin(new_df["stock"].unique())], new_df]))
    cands = [c for c in cands if c in set(new_df["stock"])]
    if cands:
        print(f"\nChecking corporate actions for {len(cands)} stocks with large moves ...")
        save_actions(fetch_actions(cands, new_df["date"].min() - timedelta(days=10)))

    # New ETFs list often; keep them out of the stock universe (picks.py excludes etf_symbols.csv)
    try:
        from etf_list import refresh_etfs
        n_etf = refresh_etfs()
        if n_etf:
            print(f"  added {n_etf} new ETFs to etf_symbols.csv")
    except Exception as e:
        print(f"  ETF list not refreshed ({type(e).__name__}); using the saved list")

    # 5. Final save
    combined = _save(df, new_rows)
    latest   = combined.groupby("stock")["date"].max()
    print(f"\nUpdated {DATA_FILE}")
    print(f"  Total rows    : {len(combined):,}")
    print(f"  Stocks        : {combined['stock'].nunique()}  (was {df['stock'].nunique()})")
    print(f"  Date range    : {combined['date'].min().date()} to {combined['date'].max().date()}")
    print(f"  Stocks up to date ({latest.max().date()}): {(latest == latest.max()).sum()}")
    print("Done. Now run picks.py for fresh picks.")


def _fetch_angel(api, token, start, to_date):
    """Fetch daily candles from Angel One, retrying on rate limits. Returns DataFrame or None."""
    params = {
        "exchange":    "NSE",
        "symboltoken": token,
        "interval":    "ONE_DAY",
        "fromdate":    start.strftime("%Y-%m-%d 09:15"),
        "todate":      to_date,
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = api.getCandleData(params)
        except Exception:
            resp = None
        if resp and resp.get("status"):
            if not resp.get("data"):
                return None
            tmp = pd.DataFrame(resp["data"], columns=["datetime", "open", "high", "low", "close", "volume"])
            tmp["date"] = pd.to_datetime(tmp["datetime"]).dt.tz_localize(None).dt.normalize()
            return tmp
        # Rate limited or network error -- back off and retry
        time.sleep(1.0 * (attempt + 1))
    return None


def _fetch_yfinance_batch(symbols, starts):
    """Batch-download symbols from yfinance. Returns ({symbol: DataFrame}, failed_symbols).
    Yahoo rate-limits large requests (HTTP 429), so symbols are fetched in small batches with
    pauses, and anything that fails is retried in later passes with longer waits."""
    import yfinance as yf
    frames, pending, events = {}, list(symbols), []
    for attempt in range(YF_PASSES):
        if not pending:
            break
        if attempt:
            wait = 30 * attempt
            print(f"  yfinance retry pass {attempt + 1}: {len(pending)} symbols, waiting {wait}s ...", flush=True)
            time.sleep(wait)
        failed = []
        for b in range(0, len(pending), YF_BATCH):
            batch   = pending[b:b + YF_BATCH]
            start   = min(starts[s] for s in batch).date()
            tickers = [s + ".NS" for s in batch]
            try:
                # auto_adjust=False: raw closes, same basis as Angel One (splits handled in clean_data)
                raw = yf.download(tickers, start=str(start), auto_adjust=False, actions=True,
                                  progress=False, group_by="ticker", threads=True)
            except Exception:
                raw = None
            for s, t in zip(batch, tickers):
                try:
                    sub = raw[t].dropna(how="all").reset_index()
                    sub = sub.rename(columns={"Date": "date", "Open": "open", "High": "high",
                                              "Low": "low", "Close": "close", "Volume": "volume"})
                    sub["date"] = pd.to_datetime(sub["date"]).dt.tz_localize(None).dt.normalize()
                    # Yahoo's "Close" is split-adjusted even with auto_adjust=False. Undo that so
                    # prices are raw like Angel One's: multiply each row by every split after it.
                    split = sub["Stock Splits"].fillna(0) if "Stock Splits" in sub else pd.Series(0.0, index=sub.index)
                    for d, r in zip(sub["date"], split):
                        if r > 0:
                            events.append({"stock": s, "date": d, "ratio": float(r)})
                    after = split.where(split > 0, 1.0)[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
                    for c in ["open", "high", "low", "close"]:
                        sub[c] = sub[c] * after
                    sub["volume"] = sub["volume"] / after
                    sub = sub[(sub["date"] >= starts[s]) & (sub["volume"] > 0)]
                    if len(sub) == 0:
                        raise ValueError
                    frames[s] = sub
                except Exception:
                    failed.append(s)
            time.sleep(YF_PAUSE)
            done = min(b + YF_BATCH, len(pending))
            if done % 500 < YF_BATCH or done == len(pending):
                print(f"  yfinance pass {attempt + 1}: {done}/{len(pending)} | ok so far {len(frames)}", flush=True)
        pending = failed
    if events:
        save_actions(pd.DataFrame(events))
        print(f"  recorded {len(events)} split/bonus events from Yahoo", flush=True)
    return frames, pending


def _finish(tmp, symbol, cutoff):
    """Compute returns within the fetched window and keep completed days only."""
    tmp = tmp.sort_values("date").drop_duplicates("date")
    tmp = tmp[tmp["date"] <= cutoff]
    tmp["return_1d"] = tmp["close"].pct_change().clip(-1.0, 1.0)
    tmp["stock"] = symbol
    tmp = tmp.dropna(subset=["return_1d"])
    return tmp[COLS] if len(tmp) > 0 else None


def _save(df, new_rows):
    new_df   = _round_df(pd.concat(new_rows, axis=0, ignore_index=True))
    combined = pd.concat([df, new_df], axis=0, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "stock"], keep="last")
    combined = clean(combined)
    combined.to_csv(DATA_FILE, index=False)
    return combined


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
