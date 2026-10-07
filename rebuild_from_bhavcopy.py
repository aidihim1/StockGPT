# rebuild_from_bhavcopy.py -- rebuilds every price from 12 Apr 2021 on from NSE's official bhavcopies
#
# Prices from 2021 had been stitched together from several broker downloads made at different times;
# Angel One back-adjusts its history for later splits, so stitched pieces sat on different price
# bases (fake one-day returns), renamed companies were stored twice, and some days were missing.
# NSE's daily bhavcopy (fetch_bhavcopy.py) is the exchange's own record: raw official OHLC and volume
# for every security traded that day. This script:
#   1. reads all bhavcopies: EQ, BE and BZ series; company shares (ISIN INE...) and the ETFs in
#      etf_symbols.csv; no rights entitlements
#   2. identifies each company by its ISIN issuer code (ISIN characters 1-7), which survives renames
#      and face-value changes, so a renamed company is one series; it is stored under its latest
#      symbol (delisted companies under their last symbol)
#   3. replaces everything from 12 Apr 2021 on with these rows (raw prices; clean() turns them into
#      returns, adjusting recorded splits/bonuses and adding dividends)
#   4. keeps the history before 12 Apr 2021 (Yahoo 2000-2019, Angel One 2020-21 gap fill), merged
#      under the same company, and rescales its price level so its last close equals NSE's previous
#      close on the first bhavcopy day (stored returns before 2021 are kept as they are)
#   5. runs clean() and saves dataset.csv (temp file first); companies that stopped trading are
#      listed in delisted_added.csv
#
# Usage:  python rebuild_from_bhavcopy.py      (after fetch_bhavcopy.py; update_prices.py runs both)

import glob
import os
import zipfile

import numpy as np
import pandas as pd

from clean_data import ANGEL_START, clean
from corporate_actions import DIV_FILE, FILE as ACTIONS_FILE, RIGHTS_FILE, load_actions, load_dividends
from etf_list import load_etfs
from update_data import COLS, DATA_FILE, _round_df

BHAV_DIR = "bhavcopy"
SERIES = {"EQ": 0, "BE": 1, "BZ": 2}
RIGHTS_PATTERN = r"-RE\d*$"
JOIN_MAX_DAYS = 10            # rescale older history only if it ends within this many days of the first bhavcopy day


def read_day(path: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    df.columns = df.columns.str.strip()
    if "SYMBOL" in df.columns:
        m = {"SYMBOL": "symbol", "SERIES": "series", "ISIN": "isin", "OPEN": "open", "HIGH": "high", "LOW": "low",
             "CLOSE": "close", "PREVCLOSE": "prev", "TOTTRDQTY": "volume"}
    else:
        m = {"TckrSymb": "symbol", "SctySrs": "series", "ISIN": "isin", "OpnPric": "open", "HghPric": "high",
             "LwPric": "low", "ClsPric": "close", "PrvsClsgPric": "prev", "TtlTradgVol": "volume"}
    df = df.rename(columns=m)[list(m.values())]
    for c in ["symbol", "series", "isin"]:
        df[c] = df[c].astype(str).str.strip()
    df["date"] = pd.Timestamp(os.path.basename(path)[:8])
    return df


def load_bhavcopies(with_rights: bool = False):
    files = sorted(glob.glob(os.path.join(BHAV_DIR, "*.csv.zip")))
    print(f"Reading {len(files)} bhavcopies ...", flush=True)
    raw = pd.concat([read_day(f) for f in files], ignore_index=True)
    etfs = load_etfs()
    rights = raw[raw["symbol"].str.contains(RIGHTS_PATTERN, regex=True) & raw["isin"].str.startswith("INE")]
    keep = (raw["series"].isin(SERIES) & (raw["volume"] > 0) & (raw["close"] > 0)
            & (raw["isin"].str.startswith("INE") | raw["symbol"].isin(etfs))
            & ~raw["symbol"].str.contains(RIGHTS_PATTERN, regex=True))
    b = raw[keep].copy()
    # company id: ISIN issuer code for companies (stable across renames and face-value changes); ETFs by
    # ISIN. An issuer with two share classes trading on the same day (e.g. ordinary and DVR shares)
    # keeps each ISIN separate.
    b["cid"] = np.where(b["isin"].str.startswith("INE"), b["isin"].str[:7], b["isin"])
    per_day = b.drop_duplicates(["date", "isin"]).groupby(["cid", "date"])["isin"].nunique()
    two_classes = set(per_day[per_day > 1].index.get_level_values(0))
    b.loc[b["cid"].isin(two_classes), "cid"] = b.loc[b["cid"].isin(two_classes), "isin"]
    b["order"] = b["series"].map(SERIES)
    b = b.sort_values(["order", "volume"], ascending=[True, False]).drop_duplicates(["date", "cid"])
    latest = b.sort_values("date").groupby("cid")["symbol"].last()            # name = latest symbol
    b["stock"] = b["cid"].map(latest)
    b = b.drop(columns="order")
    if not with_rights:
        return b
    # rights entitlements: issuer code -> the company's name, and the first day they traded
    issuer_name = b[b["isin"].str.startswith("INE")].assign(code=lambda x: x["isin"].str[:7]).groupby("code")["stock"].last()
    ev = rights.assign(code=rights["isin"].str[:7]).groupby(["code", "symbol"])["date"].min().reset_index()
    # parent: the symbol before "-RE" if it is a known share (e.g. GATECHDVR-RE -> GATECHDVR), else the issuer
    by_symbol = b.groupby("symbol")["stock"].last()
    parent = ev["symbol"].str.replace(RIGHTS_PATTERN, "", regex=True).map(by_symbol)
    ev["stock"] = parent.fillna(ev["code"].map(issuer_name))
    ev = ev.dropna(subset=["stock"]).drop_duplicates(["stock", "date"])
    return b, ev[["stock", "date", "symbol"]].sort_values(["stock", "date"])


def split_events(b: pd.DataFrame, recorded: pd.DataFrame) -> pd.DataFrame:
    """Splits seen in the bhavcopies: the ISIN changes (new face value) and the price moves by about a
    split factor. Only where no split/bonus is recorded within 3 days."""
    x = b.sort_values(["stock", "date"])
    prev_isin, prev_close = x.groupby("stock")["isin"].shift(), x.groupby("stock")["close"].shift()
    ch = x[(prev_isin.notna()) & (x["isin"] != prev_isin)].assign(r=(x["close"] / prev_close))
    factors = np.array([2, 2.5, 4, 5, 10, 20, 0.5, 0.2, 0.1])
    out = []
    for st, d, r in ch[["stock", "date", "r"]].itertuples(index=False):
        k = factors[np.argmin(np.abs(r * factors - 1))]
        if abs(r * k - 1) < 0.15:
            near = recorded[(recorded["stock"] == st) & ((recorded["date"] - d).abs() <= pd.Timedelta(days=3))]
            if near.empty:
                out.append({"stock": st, "date": d, "ratio": float(k)})
    return pd.DataFrame(out, columns=["stock", "date", "ratio"])


def main():
    print("Loading dataset ...", flush=True)
    df = pd.read_csv(DATA_FILE, parse_dates=["date"], usecols=COLS)
    b, rights = load_bhavcopies(with_rights=True)
    names = b.groupby("cid")["symbol"].unique()
    canon = b.groupby("cid")["stock"].first()
    renamed = {s: canon[c] for c, syms in names.items() for s in syms}
    n_multi = int((names.map(len) > 1).sum())
    print(f"{b['cid'].nunique():,} companies/ETFs in the bhavcopies; {n_multi} traded under more than one symbol", flush=True)

    # History before the bhavcopy period (before the first bhavcopy day), under the same company name
    start = b["date"].min()
    src = df[df["date"] < start].assign(name=lambda x: x["stock"].map(lambda s: renamed.get(s, s)))
    best = src.groupby(["name", "stock"]).size().reset_index().sort_values(0).drop_duplicates("name", keep="last")
    old = src.merge(best[["name", "stock"]], on=["name", "stock"])            # one source symbol per company
    old = old.drop(columns="stock").rename(columns={"name": "stock"})[COLS]
    print(f"Older history kept: {len(old):,} rows for {old['stock'].nunique():,} stocks "
          f"({src['stock'].nunique() - old['stock'].nunique()} duplicate symbols dropped)", flush=True)

    # Rescale older price levels so the join matches NSE's previous close on the first bhavcopy day
    first = b.sort_values("date").groupby("stock").first()
    last_old = old.sort_values("date").groupby("stock").last()
    j = last_old.join(first[["date", "prev"]], rsuffix="_bhav", how="inner")
    j = j[((j["date_bhav"] - j["date"]).dt.days <= JOIN_MAX_DAYS) & (j["prev"] > 0) & (j["close"] > 0)]
    scale = (j["prev"] / j["close"]).rename("scale")
    old = old.join(scale, on="stock")
    s = old["scale"].fillna(1.0)
    for c in ["open", "high", "low", "close"]:
        old[c] = old[c] * s
    old["volume"] = old["volume"] / s
    old = old.drop(columns="scale")
    print(f"Joined {len(scale):,} stocks' older history to the bhavcopy period", flush=True)

    # Corporate actions and dividends filed under old symbols follow the company to its current name;
    # splits Yahoo missed but the bhavcopy shows (ISIN change) are added; rights events are saved
    acts = load_actions().assign(stock=lambda x: x["stock"].map(lambda s: renamed.get(s, s)))
    acts = acts.drop_duplicates(["stock", "date"], keep="first")
    seen = split_events(b, acts)
    acts = pd.concat([acts, seen], ignore_index=True).sort_values(["stock", "date"])
    acts.to_csv(ACTIONS_FILE, index=False)
    divs = load_dividends().assign(stock=lambda x: x["stock"].map(lambda s: renamed.get(s, s)))
    divs.drop_duplicates(["stock", "date"]).sort_values(["stock", "date"]).to_csv(DIV_FILE, index=False)
    rights.to_csv(RIGHTS_FILE, index=False)
    print(f"{len(seen)} splits found from ISIN changes; {len(rights)} rights issues", flush=True)

    new = b[COLS[:-1]].assign(return_1d=np.nan)
    combined = pd.concat([old, _round_df(new)], ignore_index=True).drop_duplicates(["date", "stock"], keep="last")
    combined = clean(combined)
    tmp = DATA_FILE + ".tmp"
    combined.to_csv(tmp, index=False)
    os.replace(tmp, DATA_FILE)

    last_day = b["date"].max()
    life = b.groupby("stock")["date"].agg(["min", "max", "size"])
    gone = life[life["max"] < last_day - pd.Timedelta(days=30)]
    gone.to_csv("delisted_added.csv")
    dropped = sorted(set(df.loc[df["date"] >= ANGEL_START, "stock"]) - set(b["stock"]) - set(renamed))
    print(f"Saved {DATA_FILE}: {len(combined):,} rows, {combined['stock'].nunique():,} symbols, "
          f"{combined['date'].min().date()} to {combined['date'].max().date()}")
    print(f"  {len(gone)} companies stopped trading before {last_day.date()} (delisted_added.csv)")
    print(f"  {len(dropped)} symbols from the old data are not in the bhavcopies (not EQ/BE/BZ shares): {dropped[:25]}")


if __name__ == "__main__":
    main()
