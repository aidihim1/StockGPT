# fill_gap_2020.py -- fills the January 2020 - April 2021 gap with Angel One's price history
#
# The history was built from Yahoo (2000-2019) and from 12 Apr 2021 on (Angel One, now NSE bhavcopies),
# leaving no data for Jan 2020 - Apr 2021, including the COVID crash. Angel One's API serves older daily
# candles too, but on its own basis: past prices are scaled for later splits and bonus issues. So, per
# listed stock:
#   - one request for daily candles from Nov 2019 to Jun 2021, saved in gap_cache/ (a re-run only
#     fetches what is missing; the login is renewed when the session expires, e.g. at midnight)
#   - returns inside the gap come from Angel One's own consecutive closes (consistent with each other);
#     a split Angel One did not adjust (an exact split ratio) is blanked, as clean() does
#   - prices are put on the stored basis using the overlap right after the gap (else right before it)
#   - only days missing between 1 Jan 2020 and the stock's first stored day after that are added;
#     nothing already stored changes. clean() then links the first stored day to the filled days.
# Companies no longer listed have no Angel One token and keep their bhavcopy history only.
# Run once; the regular update (update_prices.py) keeps this history as it is.
#
# Usage:  python fill_gap_2020.py            (needs config.py with Angel One credentials)

import logging
import os
import time

import numpy as np
import pandas as pd

from clean_data import BAND, SUSPECT_FACTORS, clean
from update_data import ANGEL_HOST, COLS, DATA_FILE, MAX_RETRIES, SLEEP_BETWEEN, _reachable, _round_df

GAP_START = pd.Timestamp("2020-01-01")
FETCH_FROM, FETCH_TO = "2019-11-01 09:15", "2021-06-30 15:30"
ANCHOR_DAYS = 10                    # overlap days used to find the price scale
CACHE = "gap_cache"
SESSION_ERRORS = {"AG8001", "AG8002", "AG8003", "AB1010"}   # invalid / expired session token
CANDLE_COLS = ["datetime", "open", "high", "low", "close", "volume"]


def login():
    import logzero
    logzero.loglevel(logging.CRITICAL)   # the SmartAPI library logs request headers (API key) on errors
    from login import get_api
    return get_api()


def fetch(api_box: list, token: str):
    """Daily candles for the gap window, or None if the request keeps failing. Logs in again when the
    session has expired."""
    params = {"exchange": "NSE", "symboltoken": token, "interval": "ONE_DAY",
              "fromdate": FETCH_FROM, "todate": FETCH_TO}
    for attempt in range(MAX_RETRIES):
        try:
            resp = api_box[0].getCandleData(params)
        except Exception:
            resp = None
        if resp and resp.get("status"):
            return pd.DataFrame(resp.get("data") or [], columns=CANDLE_COLS)
        if resp and resp.get("errorCode") in SESSION_ERRORS:
            print("  session expired, logging in again", flush=True)
            api_box[0] = login()
            continue
        time.sleep(1.0 * (attempt + 1))
    return None


def gap_rows(a: pd.DataFrame, stored: pd.Series):
    """Rows to add for one stock (prices on the stored basis), or None."""
    a = a.assign(date=pd.to_datetime(a["datetime"]).dt.tz_localize(None).dt.normalize())
    a = a.sort_values("date").drop_duplicates("date").set_index("date")
    a = a[a["volume"] > 0]
    if a.empty:
        return None
    r = a["close"].pct_change()
    near_factor = np.min([np.abs((1 + r) * k - 1) for k in SUSPECT_FACTORS], axis=0) < 0.01
    r[((r < BAND[0]) | (r > BAND[1])) & near_factor] = np.nan
    a["return_1d"] = r
    after = stored.index[stored.index >= GAP_START]
    first_after = after.min() if len(after) else pd.Timestamp.max
    ratio = (stored.reindex(a.index) / a["close"]).dropna()
    post, pre = ratio[ratio.index >= first_after].head(ANCHOR_DAYS), ratio[ratio.index < GAP_START].tail(ANCHOR_DAYS)
    scale = post.median() if len(post) else (pre.median() if len(pre) else np.nan)
    gap = a[(a.index >= GAP_START) & (a.index < first_after)].copy()
    if np.isnan(scale) or gap.empty:
        return None
    for c in ["open", "high", "low", "close"]:
        gap[c] = gap[c] * scale
    gap["volume"] = gap["volume"] / scale
    return gap.reset_index()


def main():
    if not _reachable(ANGEL_HOST):
        raise SystemExit(f"{ANGEL_HOST} is not reachable; try again later.")
    os.makedirs(CACHE, exist_ok=True)
    print("Loading dataset ...", flush=True)
    df = pd.read_csv(DATA_FILE, parse_dates=["date"], usecols=COLS)
    sym = pd.read_csv("nse_symbols.csv").drop_duplicates("clean_symbol")
    sym = sym[sym["clean_symbol"].isin(set(df["stock"]))]
    todo = [(str(t), s) for t, s in zip(sym["token"], sym["clean_symbol"])
            if not os.path.exists(os.path.join(CACHE, f"{s}.csv"))]
    print(f"{len(sym)} stocks; {len(todo)} to fetch (Nov 2019 - Jun 2021)", flush=True)

    api_box, failed = [login()], []
    for i, (token, s) in enumerate(todo, 1):
        a = fetch(api_box, token)
        time.sleep(SLEEP_BETWEEN)
        if a is None:
            failed.append(s)
        else:
            a.to_csv(os.path.join(CACHE, f"{s}.csv"), index=False)      # empty file = no data for this stock
        if i % 250 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} fetched | failed {len(failed)}", flush=True)
    if failed:
        print(f"  could not fetch {len(failed)} stocks (re-run to retry): {failed[:20]}")

    stored = {s: g.set_index("date")["close"] for s, g in df[["date", "stock", "close"]].groupby("stock")}
    rows, n_none = [], 0
    for s in sym["clean_symbol"]:
        f = os.path.join(CACHE, f"{s}.csv")
        if not os.path.exists(f):
            continue
        a = pd.read_csv(f)
        g = gap_rows(a, stored[s]) if len(a) else None
        if g is None:
            n_none += 1
            continue
        g["stock"] = s
        rows.append(g[COLS])
    new = _round_df(pd.concat(rows, ignore_index=True))
    print(f"Adding {len(new):,} rows for {len(rows)} stocks ({new['date'].min().date()} to {new['date'].max().date()}); "
          f"{n_none} stocks have nothing to add (not listed then)", flush=True)
    combined = pd.concat([df, new], ignore_index=True).drop_duplicates(["date", "stock"], keep="first")
    combined = clean(combined)
    tmp = DATA_FILE + ".tmp"
    combined.to_csv(tmp, index=False)
    os.replace(tmp, DATA_FILE)
    print(f"Saved {DATA_FILE}: {len(combined):,} rows")


if __name__ == "__main__":
    main()
