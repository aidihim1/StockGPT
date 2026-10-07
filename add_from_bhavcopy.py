# add_from_bhavcopy.py -- removes survivorship bias and fills missing days, using NSE bhavcopy files
#
# The price history came from Angel One's list of currently listed stocks, so companies that were
# delisted or merged before 2026 were missing (survivorship bias). NSE's daily bhavcopy files
# (fetch_bhavcopy.py) list every company traded each day. This script:
#   1. reads all bhavcopies in bhavcopy/: EQ, BE and BZ series, company shares only (ISIN INE...,
#      not rights entitlements)
#   2. maps old symbols to current ones (NSE's symbol-change list, and ISINs of listed companies),
#      so renamed companies are not added twice
#   3. adds the companies that are still missing: their bhavcopy history (exchange prices), plus
#      2000-2019 Yahoo history when Yahoo still has it, and their recorded splits/dividends
#   4. fills days missing for stocks already in the data (feed gaps, special sessions, vendor
#      misses), rescaled to the stored price basis (Angel One sometimes back-adjusts for later
#      splits): only inside a stock's history and only where the stored/bhavcopy price ratio is
#      the same on the traded days before and after the gap
#   5. runs clean() and saves dataset.csv (temp file first)
# Run once while the data was being built. rebuild_from_bhavcopy.py (run by update_prices.py) now
# rebuilds every price from 12 Apr 2021 on from the bhavcopies, so this is not part of the regular update.
#
# Usage:  python add_from_bhavcopy.py           (after fetch_bhavcopy.py; --no_yahoo skips the Yahoo
#                                                history and splits/dividends of the added companies)

import argparse
import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd
import requests

from clean_data import ANGEL_START, clean
from corporate_actions import fetch_actions, save_actions, save_dividends
from etf_list import HEADERS, NSE_EQUITY_URL
from update_data import COLS, DATA_FILE, _round_df

BHAV_DIR = "bhavcopy"
SYMBOL_CHANGE_URL = "https://nsearchives.nseindia.com/content/equities/symbolchange.csv"
SERIES = {"EQ": 0, "BE": 1, "BZ": 2}
SCALE_TOL = 0.01
RIGHTS_PATTERN = r"-RE\d*$"         # rights entitlements (e.g. SUZLON-RE) trade for a few days during a rights issue


def read_bhavcopy(path: str) -> pd.DataFrame:
    """One day's bhavcopy (old or UDiFF format) -> date, stock, series, isin, open, high, low, close, volume."""
    with zipfile.ZipFile(path) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    df.columns = df.columns.str.strip()
    if "SYMBOL" in df.columns:
        df = df.rename(columns={"SYMBOL": "stock", "SERIES": "series", "ISIN": "isin", "OPEN": "open",
                                "HIGH": "high", "LOW": "low", "CLOSE": "close", "TOTTRDQTY": "volume"})
    else:
        df = df.rename(columns={"TckrSymb": "stock", "SctySrs": "series", "ISIN": "isin", "OpnPric": "open",
                                "HghPric": "high", "LwPric": "low", "ClsPric": "close", "TtlTradgVol": "volume"})
    df = df[["stock", "series", "isin", "open", "high", "low", "close", "volume"]].copy()
    for c in ["stock", "series", "isin"]:
        df[c] = df[c].astype(str).str.strip()
    df["date"] = pd.Timestamp(os.path.basename(path)[:8])
    return df


def load_bhavcopies() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(BHAV_DIR, "*.csv.zip")))
    print(f"Reading {len(files)} bhavcopies ...", flush=True)
    b = pd.concat([read_bhavcopy(f) for f in files], ignore_index=True)
    b = b[b["series"].isin(SERIES) & b["isin"].str.startswith("INE") & (b["volume"] > 0) & (b["close"] > 0)
          & ~b["stock"].str.contains(RIGHTS_PATTERN, regex=True)]     # rights entitlements are not shares
    b["order"] = b["series"].map(SERIES)
    return b.sort_values("order").drop_duplicates(["date", "stock"]).drop(columns="order")


def symbol_map(b: pd.DataFrame, have: set) -> pd.Series:
    """For every (row) symbol in b: the symbol the company has in our data, or its own symbol."""
    sc = pd.read_csv(io.StringIO(requests.get(SYMBOL_CHANGE_URL, headers=HEADERS, timeout=60).text),
                     header=None, encoding="latin-1").iloc[:, :4]
    sc.columns = ["name", "old", "new", "date"]
    sc["old"], sc["new"] = sc["old"].astype(str).str.strip(), sc["new"].astype(str).str.strip()
    sc["date"] = pd.to_datetime(sc["date"].astype(str).str.strip(), format="%d-%b-%Y", errors="coerce")
    nxt = sc.sort_values("date").drop_duplicates("old", keep="last").set_index("old")
    eq = pd.read_csv(io.StringIO(requests.get(NSE_EQUITY_URL, headers=HEADERS, timeout=60).text))
    eq.columns = eq.columns.str.strip()
    by_isin = dict(zip(eq["ISIN NUMBER"].str.strip(), eq["SYMBOL"].str.strip()))

    def follow(sym, day):
        for _ in range(10):                       # chains of renames, each after this row's date
            if sym in have or sym not in nxt.index or not nxt.at[sym, "date"] > day:
                break
            sym = nxt.at[sym, "new"]
        return sym

    keys = b[["stock", "isin", "date"]].drop_duplicates(["stock", "isin"], keep="last")
    out = {}
    for s, isin, day in keys.itertuples(index=False):
        m = follow(s, day)
        if m not in have and by_isin.get(isin) in have:
            m = by_isin[isin]
        out[(s, isin)] = m
    return pd.Series([out[k] for k in zip(b["stock"], b["isin"])], index=b.index)


def fill_gaps(df: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Bhavcopy rows for days missing inside existing stocks' histories, on the stored price scale."""
    have = df.loc[df["date"] >= ANGEL_START, ["date", "stock", "close"]]
    span = df.groupby("stock")["date"].agg(["min", "max"])
    x = b[b["stock"].isin(span.index)].merge(have, on=["date", "stock"], how="left", suffixes=("", "_stored"))
    x = x.sort_values(["stock", "date"])
    x["scale"] = x["close_stored"] / x["close"]
    g = x.groupby("stock")["scale"]
    x["before"], x["after"] = g.ffill(), g.bfill()
    lo, hi = x["stock"].map(span["min"]), x["stock"].map(span["max"])
    fill = (x["close_stored"].isna() & (x["date"] > lo) & (x["date"] < hi) &
            ((x["before"] / x["after"] - 1).abs() < SCALE_TOL))
    f = x[fill].copy()
    for c in ["open", "high", "low", "close"]:
        f[c] = f[c] * f["before"]
    f["volume"] = f["volume"] / f["before"]
    f["return_1d"] = np.nan                       # recomputed by clean()
    return f[COLS]


def main(args):
    print("Loading dataset ...", flush=True)
    df = pd.read_csv(DATA_FILE, parse_dates=["date"], usecols=COLS)
    have = set(df["stock"])
    b = load_bhavcopies()
    b = b[b["date"] <= df["date"].max()]
    b["stock"] = symbol_map(b, have)

    gaps = fill_gaps(df, b)
    by_date = gaps.groupby("date").size().sort_values(ascending=False)
    print(f"Filling {len(gaps):,} missing stock-days for {gaps['stock'].nunique():,} existing stocks "
          f"(largest days: {', '.join(f'{d.date()} {n}' for d, n in by_date.head(6).items())})", flush=True)

    missing = b[~b["stock"].isin(have)].copy()
    last_day = b["date"].max()
    life = missing.groupby("stock")["date"].agg(["min", "max", "size"])
    print(f"Adding {len(life):,} companies missing from the data ({len(missing):,} rows); "
          f"{int((life['max'] < last_day - pd.Timedelta(days=30)).sum())} stopped trading before {last_day.date()}", flush=True)
    missing = missing.sort_values(["stock", "date"])
    missing["return_1d"] = missing.groupby("stock")["close"].pct_change()
    added = missing[COLS]

    if not args.no_yahoo and len(life):
        from add_symbols import old_history
        old_syms = life.index[life["min"] < ANGEL_START + pd.Timedelta(days=30)].tolist()
        print(f"Looking for 2000-2019 Yahoo history for {len(old_syms)} of them ...", flush=True)
        old = old_history(old_syms)
        print(f"  got {len(old)}", flush=True)
        if old:
            added = pd.concat([added] + list(old.values()), ignore_index=True)
        splits, divs = fetch_actions(list(life.index), ANGEL_START - pd.Timedelta(days=10), with_dividends=True)
        save_actions(splits)
        save_dividends(divs)
        print(f"  recorded {len(splits)} splits/bonuses and {len(divs)} dividends for them", flush=True)

    combined = pd.concat([df, _round_df(pd.concat([gaps, added], ignore_index=True))], ignore_index=True)
    combined = combined.drop_duplicates(["date", "stock"], keep="first")
    combined = clean(combined)
    tmp = DATA_FILE + ".tmp"
    combined.to_csv(tmp, index=False)
    os.replace(tmp, DATA_FILE)
    life.to_csv("delisted_added.csv")
    print(f"Saved {DATA_FILE}: {len(combined):,} rows, {combined['stock'].nunique():,} symbols "
          f"(list of added companies: delisted_added.csv)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no_yahoo", action="store_true", help="skip Yahoo history/actions for added companies")
    main(p.parse_args())
