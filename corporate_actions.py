# corporate_actions.py -- recorded stock splits / bonus issues and dividends, used by clean_data.py
#
# Raw exchange prices (NSE bhavcopies) are not adjusted for splits or bonus issues, so the day a stock
# splits 1:10 looks like a -90% crash. Guessing splits from the price drop alone is unsafe: a -20%
# lower circuit or a -28% crash can look like a 1:2 bonus (x2/3). So a return is only adjusted when a
# split/bonus is recorded for that stock within a few days of the move: by Yahoo Finance, or as an
# ISIN change in the bhavcopies (added by rebuild_from_bhavcopy.py).
#
# corporate_actions.csv columns: stock, date, ratio  (ratio = new shares per old share,
#   e.g. 2.0 for a 1:1 bonus or a 2-for-1 split, 1.2 for a 1:5 bonus; price falls by 1/ratio that day)
# dividends.csv columns: stock, date, yield  (cash dividend / previous close, on the ex-date; a yield
#   does not depend on how prices are split-adjusted). Prices from 2021 on are price-only, so
#   clean_data adds these yields to make total returns, like the pre-2020 history.
# demergers.csv columns: stock, date, note  (ex-dates, a hand-checked list)
# rights_events.csv columns: stock, date, symbol  (first day the rights entitlements traded, from the
#   bhavcopies). clean_data uses both to blank the ex-date fall, which is not a loss.
#
# Usage:  python corporate_actions.py                    # splits for every stock with a big move since 2021
#         python corporate_actions.py --all              # splits and dividends for every company stock
#         python corporate_actions.py --all --days 400   # the same for the last 400 days (update_prices.py)

import argparse
import os
import time

import pandas as pd

FILE = "corporate_actions.csv"
DIV_FILE = "dividends.csv"
ANGEL_START = pd.Timestamp("2021-04-12")
BATCH, PAUSE, PASSES = 25, 2, 4
MAX_YIELD = 0.25          # larger "dividends" are data errors or special payouts handled elsewhere


def load_actions() -> pd.DataFrame:
    if not os.path.exists(FILE):
        return pd.DataFrame(columns=["stock", "date", "ratio"])
    return pd.read_csv(FILE, parse_dates=["date"])


def save_actions(new: pd.DataFrame):
    allx = pd.concat([load_actions(), new], ignore_index=True)
    allx["date"] = pd.to_datetime(allx["date"])
    allx = allx.drop_duplicates(["stock", "date"], keep="last").sort_values(["stock", "date"])
    allx.to_csv(FILE, index=False)
    return allx


DEMERGER_FILE = "demergers.csv"


def load_demergers() -> pd.DataFrame:
    """demergers.csv: stock, date (ex-date), note -- a hand-checked list; Yahoo does not record demergers."""
    if not os.path.exists(DEMERGER_FILE):
        return pd.DataFrame(columns=["stock", "date", "note"])
    return pd.read_csv(DEMERGER_FILE, parse_dates=["date"])


RIGHTS_FILE = "rights_events.csv"


def load_rights() -> pd.DataFrame:
    """rights_events.csv: stock, date the rights entitlements started trading (from the bhavcopies)."""
    if not os.path.exists(RIGHTS_FILE):
        return pd.DataFrame(columns=["stock", "date"])
    return pd.read_csv(RIGHTS_FILE, parse_dates=["date"])


def load_dividends() -> pd.DataFrame:
    if not os.path.exists(DIV_FILE):
        return pd.DataFrame(columns=["stock", "date", "yield"])
    return pd.read_csv(DIV_FILE, parse_dates=["date"])


def save_dividends(new: pd.DataFrame):
    allx = pd.concat([load_dividends(), new], ignore_index=True)
    allx["date"] = pd.to_datetime(allx["date"])
    allx = allx.drop_duplicates(["stock", "date"], keep="last").sort_values(["stock", "date"])
    allx.to_csv(DIV_FILE, index=False)
    return allx


def actions_from_download(raw, tickers):
    """(splits, dividends) from a yf.download(..., actions=True, auto_adjust=False, group_by='ticker') frame.
    Yahoo's dividends and closes are both split-adjusted, so dividend / previous close is a true yield."""
    splits, divs = [], []
    for t in tickers:
        try:
            sub = raw[t].dropna(how="all")
        except Exception:
            continue
        stock = t[:-3]
        sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
        if "Stock Splits" in sub:
            s = sub["Stock Splits"].fillna(0)
            for d, r in s[s > 0].items():
                splits.append({"stock": stock, "date": d, "ratio": float(r)})
        if "Dividends" in sub and "Close" in sub:
            prev = sub["Close"].shift(1)
            dv = sub["Dividends"].fillna(0)
            for d, a in dv[dv > 0].items():
                y = a / prev.loc[d] if prev.loc[d] > 0 else float("nan")
                if 0 < y < MAX_YIELD:
                    divs.append({"stock": stock, "date": d, "yield": round(float(y), 6)})
    return (pd.DataFrame(splits, columns=["stock", "date", "ratio"]),
            pd.DataFrame(divs, columns=["stock", "date", "yield"]))


def splits_from_download(raw, tickers) -> pd.DataFrame:
    return actions_from_download(raw, tickers)[0]


def fetch_actions(symbols, start, with_dividends=False):
    """Download split/bonus (and optionally dividend) history from Yahoo in small batches, retrying
    symbols that fail (rate limits). Returns splits, or (splits, dividends) with with_dividends=True."""
    import yfinance as yf
    sp, dv, pending = [], [], list(symbols)
    for attempt in range(PASSES):
        if not pending:
            break
        if attempt:
            time.sleep(30 * attempt)
        failed = []
        for b in range(0, len(pending), BATCH):
            batch = pending[b:b + BATCH]
            tickers = [s + ".NS" for s in batch]
            try:
                raw = yf.download(tickers, start=str(pd.Timestamp(start).date()), auto_adjust=False,
                                  actions=True, progress=False, group_by="ticker", threads=True)
            except Exception:
                raw = None
            ok_t = []
            for s, t in zip(batch, tickers):
                try:
                    ok = raw is not None and raw[t]["Close"].notna().any()
                except Exception:
                    ok = False
                (ok_t if ok else failed).append(t if ok else s)
            if raw is not None and ok_t:
                a, d = actions_from_download(raw, ok_t)
                sp.append(a)
                dv.append(d)
            time.sleep(PAUSE)
        print(f"  actions pass {attempt + 1}: {len(pending) - len(failed)}/{len(pending)} fetched", flush=True)
        pending = failed
    if pending:
        print(f"  could not fetch actions for {len(pending)} stocks: {', '.join(pending[:20])}")
    splits = pd.concat(sp, ignore_index=True) if sp else pd.DataFrame(columns=["stock", "date", "ratio"])
    divs = pd.concat(dv, ignore_index=True) if dv else pd.DataFrame(columns=["stock", "date", "yield"])
    return (splits, divs) if with_dividends else splits


def candidate_stocks(df: pd.DataFrame, band=(-0.20, 0.25)) -> list:
    """Stocks with at least one raw daily move outside the circuit band since ANGEL_START."""
    d = df[df["date"] >= ANGEL_START - pd.Timedelta(days=10)].sort_values(["stock", "date"])
    raw = d["close"] / d.groupby("stock")["close"].shift() - 1
    return sorted(d.loc[(raw < band[0]) | (raw > band[1]), "stock"].unique())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="every company stock (splits and dividends), not just big movers")
    p.add_argument("--days", type=int, default=None, help="with --all: only the last N days (quick refresh)")
    a = p.parse_args()
    data = pd.read_csv("dataset.csv", parse_dates=["date"], usecols=["date", "stock", "close"])
    if a.all:
        from etf_list import load_etfs
        etfs = load_etfs()
        cands = sorted(s for s in data.loc[data["date"] >= ANGEL_START, "stock"].unique() if s not in etfs)
        print(f"Fetching splits and dividends since {ANGEL_START.date()} for {len(cands)} company stocks")
        since = pd.Timestamp.now().normalize() - pd.Timedelta(days=a.days) if a.days else ANGEL_START - pd.Timedelta(days=10)
        if a.days:     # only stocks still trading recently
            cands = sorted(set(cands) & set(data.loc[data["date"] >= since, "stock"]))
        splits, divs = fetch_actions(cands, since, with_dividends=True)
        alld = save_dividends(divs)
        print(f"{len(divs)} dividends found; {DIV_FILE} now has {len(alld)}")
    else:
        cands = candidate_stocks(data)
        print(f"{len(cands)} stocks with moves beyond the circuit band since {ANGEL_START.date()}")
        splits = fetch_actions(cands, ANGEL_START - pd.Timedelta(days=10))
    allx = save_actions(splits)
    print(f"{len(splits)} split/bonus events found; {FILE} now has {len(allx)} events")
