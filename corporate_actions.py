# corporate_actions.py -- recorded stock splits / bonus issues, used by clean_data.py
#
# Angel One prices are not adjusted for splits or bonus issues, so the day a stock splits 1:10
# looks like a -90% crash. Guessing splits from the price drop alone is unsafe: a -20% lower
# circuit or a -28% crash can look like a 1:2 bonus (x2/3). So a return is only adjusted when
# Yahoo Finance records a split/bonus for that stock within a few days of the move.
#
# corporate_actions.csv columns: stock, date, ratio  (ratio = new shares per old share,
# e.g. 2.0 for a 1:1 bonus or a 2-for-1 split; the price falls by 1/ratio on that day)
#
# Usage:  python corporate_actions.py     # fetch actions for every stock with a big move

import os
import time

import pandas as pd

FILE = "corporate_actions.csv"
ANGEL_START = pd.Timestamp("2021-04-12")
BATCH, PAUSE, PASSES = 25, 2, 4


def load_actions() -> pd.DataFrame:
    if not os.path.exists(FILE):
        return pd.DataFrame(columns=["stock", "date", "ratio"])
    return pd.read_csv(FILE, parse_dates=["date"])


def save_actions(new: pd.DataFrame):
    allx = pd.concat([load_actions(), new], ignore_index=True)
    allx = allx.drop_duplicates(["stock", "date"], keep="last").sort_values(["stock", "date"])
    allx.to_csv(FILE, index=False)
    return allx


def splits_from_download(raw, tickers) -> pd.DataFrame:
    """Extract split/bonus events from a yf.download(..., actions=True, group_by='ticker') frame."""
    rows = []
    for t in tickers:
        try:
            s = raw[t]["Stock Splits"]
        except Exception:
            continue
        s = s[s.fillna(0) > 0]
        for d, r in s.items():
            rows.append({"stock": t[:-3], "date": pd.Timestamp(d).tz_localize(None).normalize(), "ratio": float(r)})
    return pd.DataFrame(rows, columns=["stock", "date", "ratio"])


def fetch_actions(symbols, start) -> pd.DataFrame:
    """Download split/bonus history for symbols from Yahoo (small batches, retries for rate limits)."""
    import yfinance as yf
    found, pending = [], list(symbols)
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
            for s, t in zip(batch, tickers):
                try:
                    ok = raw is not None and raw[t]["Close"].notna().any()
                except Exception:
                    ok = False
                if not ok:
                    failed.append(s)
            if raw is not None:
                found.append(splits_from_download(raw, [t for s, t in zip(batch, tickers) if s not in failed]))
            time.sleep(PAUSE)
        print(f"  actions pass {attempt + 1}: {len(pending) - len(failed)}/{len(pending)} fetched", flush=True)
        pending = failed
    if pending:
        print(f"  could not fetch actions for {len(pending)} stocks: {', '.join(pending[:20])}")
    return pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=["stock", "date", "ratio"])


def candidate_stocks(df: pd.DataFrame, band=(-0.20, 0.25)) -> list:
    """Stocks with at least one raw daily move outside the circuit band since ANGEL_START."""
    d = df[df["date"] >= ANGEL_START - pd.Timedelta(days=10)].sort_values(["stock", "date"])
    raw = d["close"] / d.groupby("stock")["close"].shift() - 1
    return sorted(d.loc[(raw < band[0]) | (raw > band[1]), "stock"].unique())


if __name__ == "__main__":
    data = pd.read_csv("dataset.csv", parse_dates=["date"], usecols=["date", "stock", "close"])
    cands = candidate_stocks(data)
    print(f"{len(cands)} stocks with moves beyond the circuit band since {ANGEL_START.date()}")
    acts = fetch_actions(cands, ANGEL_START - pd.Timedelta(days=10))
    allx = save_actions(acts)
    print(f"{len(acts)} split/bonus events found; {FILE} now has {len(allx)} events")
