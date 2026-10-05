# add_symbols.py -- add NSE-listed companies that are missing from dataset.csv
#
# The symbol list from Angel One kept only "-EQ" symbols, which leaves out companies in NSE's BE/BZ
# series (trade-for-trade: stocks under surveillance or not complying with listing rules) and very
# new listings. BE/BZ stocks are often ones that did badly, so leaving them out of the history flatters
# backtests, like survivorship bias. This script:
#   1. reads NSE's official equity list (EQUITY_L.csv) and finds companies not in dataset.csv
#   2. adds them to nse_symbols.csv with their Angel One token, so update_data.py keeps them current
#   3. downloads their history from Yahoo, prepared like the rest of the data:
#        2000-2019: split- and dividend-adjusted closes (as fetch_old_data.py)
#        Apr 2021 on: raw closes, splits recorded in corporate_actions.csv (as Angel One + update_data.py)
#        nothing in between (the same Jan 2020 - Apr 2021 gap as every other stock)
#      and stops at the dataset's last date, so every stock ends on the same day
#   4. runs clean() on the combined dataset and saves it (written to a temp file first)
#
# Usage:  python add_symbols.py            (python add_symbols.py --dry_run to only list them)

import argparse
import io
import os
import time

import pandas as pd
import requests

from clean_data import clean
from etf_list import HEADERS
from update_data import COLS, DATA_FILE, _fetch_yfinance_batch, _finish, _round_df

NSE_EQUITY_URL   = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
ANGEL_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
OLD_END     = pd.Timestamp("2020-01-01")   # Yahoo adjusted history used before this date
ANGEL_START = pd.Timestamp("2021-04-12")   # raw prices from this date
BATCH, PAUSE, PASSES = 25, 2, 3
SERIES_ORDER = {"EQ": 0, "BE": 1, "BZ": 2}


def nse_equity_list() -> pd.DataFrame:
    resp = requests.get(NSE_EQUITY_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    eq = pd.read_csv(io.StringIO(resp.text))
    eq.columns = eq.columns.str.strip()
    eq = eq.rename(columns={"SYMBOL": "stock", "SERIES": "series", "DATE OF LISTING": "listed"})
    eq["stock"], eq["series"] = eq["stock"].str.strip(), eq["series"].str.strip()
    eq["listed"] = pd.to_datetime(eq["listed"].str.strip(), format="%d-%b-%Y")
    return eq[["stock", "series", "listed"]]


def angel_rows(symbols) -> pd.DataFrame:
    """nse_symbols.csv rows (Angel One token) for symbols in any of the EQ/BE/BZ series."""
    m = pd.DataFrame(requests.get(ANGEL_MASTER_URL, timeout=180).json())
    m = m[(m["exch_seg"] == "NSE") & m["symbol"].str.contains(r"-(?:EQ|BE|BZ)$")].copy()
    m["clean_symbol"] = m["symbol"].str.replace(r"-(?:EQ|BE|BZ)$", "", regex=True)
    m = m[m["clean_symbol"].isin(symbols)]
    m["order"] = m["symbol"].str[-2:].map(SERIES_ORDER)
    m = m.sort_values("order").drop_duplicates("clean_symbol")
    return m[["token", "symbol", "name", "exch_seg", "clean_symbol"]]


def old_history(symbols) -> dict:
    """2000-2019 daily history, split- and dividend-adjusted (same basis as fetch_old_data.py)."""
    import yfinance as yf
    out, pending = {}, list(symbols)
    for attempt in range(PASSES):
        if not pending:
            break
        if attempt:
            time.sleep(30 * attempt)
        failed = []
        for b in range(0, len(pending), BATCH):
            batch = pending[b:b + BATCH]
            try:
                raw = yf.download([s + ".NS" for s in batch], start="2000-01-01", end=str(OLD_END.date()),
                                  auto_adjust=True, progress=False, group_by="ticker", threads=True)
            except Exception:
                failed += batch
                continue
            for s in batch:
                try:
                    sub = raw[s + ".NS"].dropna(how="all")
                except Exception:
                    sub = pd.DataFrame()
                if sub.empty:                             # listed before 2020, so empty = failed request
                    failed.append(s)
                    continue
                if len(sub) <= 200:                       # as fetch_old_data.py
                    continue
                sub = sub.reset_index().rename(columns={"Date": "date", "Open": "open", "High": "high",
                                                        "Low": "low", "Close": "close", "Volume": "volume"})
                sub["date"] = pd.to_datetime(sub["date"]).dt.tz_localize(None).dt.normalize()
                sub = sub.sort_values("date").drop_duplicates("date")
                sub["return_1d"] = sub["close"].pct_change().clip(-1.0, 1.0)
                sub["stock"] = s
                out[s] = sub.dropna(subset=["return_1d"])[COLS]
            time.sleep(PAUSE)
        pending = failed
    if pending:
        print(f"  no 2000-2019 data on Yahoo for {len(pending)}: {pending[:20]}")
    return out


def main(args):
    print("Loading dataset ...")
    df = pd.read_csv(DATA_FILE, parse_dates=["date"], usecols=COLS)
    last_day = df["date"].max()
    eq = nse_equity_list()
    missing = eq[~eq["stock"].isin(set(df["stock"]))].sort_values("stock")
    print(f"NSE lists {len(eq):,} companies; {len(missing)} are not in {DATA_FILE} "
          f"(series {missing['series'].value_counts().to_dict()})")
    if args.dry_run or missing.empty:
        print(missing.to_string(index=False))
        return

    sym = pd.read_csv("nse_symbols.csv")
    rows = angel_rows(set(missing["stock"]) - set(sym["clean_symbol"]))
    if len(rows):
        pd.concat([sym, rows], ignore_index=True).to_csv("nse_symbols.csv", index=False)
    print(f"Added {len(rows)} symbols with Angel One tokens to nse_symbols.csv")

    symbols = missing["stock"].tolist()
    old_syms = missing.loc[missing["listed"] < OLD_END, "stock"].tolist()
    print(f"\nDownloading 2000-2019 history for {len(old_syms)} stocks listed before 2020 ...")
    old = old_history(old_syms)
    print(f"  got {len(old)}")
    print(f"Downloading history from {ANGEL_START.date()} for {len(symbols)} stocks ...")
    recent, failed = _fetch_yfinance_batch(symbols, {s: ANGEL_START for s in symbols})
    recent = {s: _finish(t, s, last_day) for s, t in recent.items()}
    recent = {s: t for s, t in recent.items() if t is not None}
    print(f"  got {len(recent)}; no data on Yahoo: {len(failed)} {failed[:20]}")

    new = pd.concat(list(old.values()) + list(recent.values()), ignore_index=True)
    added = sorted(set(new["stock"]))
    combined = pd.concat([df, _round_df(new)], ignore_index=True).drop_duplicates(["date", "stock"], keep="first")
    combined = clean(combined)
    tmp = DATA_FILE + ".tmp"
    combined.to_csv(tmp, index=False)
    os.replace(tmp, DATA_FILE)
    print(f"\nAdded {len(added)} stocks, {len(new):,} rows. {DATA_FILE}: {len(combined):,} rows, "
          f"{combined['stock'].nunique():,} symbols, {combined['date'].min().date()} to {combined['date'].max().date()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    main(p.parse_args())
