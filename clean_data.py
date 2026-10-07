# clean_data.py -- consistency fixes for dataset.csv
#
# rebuild_from_bhavcopy.py (run by update_prices.py) and the other data scripts run clean() after
# every change. Can also be run on its own:
#   python clean_data.py
#
# From ANGEL_START (12 Apr 2021) on, prices are raw exchange prices (NSE bhavcopies; Angel One or
# Yahoo with auto_adjust=False if update_data.py is used). clean():
#   1. drops dummy test symbols (e.g. 011NSETEST) and no-trade rows (volume 0; before ANGEL_START only
#      those with an unchanged price) -- data vendors emit these on NSE holidays and for untraded
#      stocks, which would feed fake flat days to the model; widens high/low where the open or close
#      falls outside them
#   2. drops "sparse" dates from ANGEL_START on: days with fewer than half the usual number of
#      stocks (special sessions such as Muhurat or Saturday sessions that the feed barely covers);
#      returns then span them, like a holiday
#   3. recomputes return_1d = close / previous close - 1 from ANGEL_START on; a return across a
#      gap of more than 10 days (a suspension) is blanked, as it is not a 1-day return
#   4. neutralises splits / bonus issues of any size: on one day per event (the ex-date, else the
#      nearest day within ACTION_WINDOW_DAYS), the return is replaced by the split-adjusted return
#      if corporate_actions.csv records the event and the price moved accordingly (never guessed
#      from the size of the move alone)
#   5. blanks big moves whose price ratio is exactly a common split factor but nothing is recorded
#   6. adds recorded cash dividends (dividends.csv, as a yield) on the ex-date, making total
#      returns like the pre-2020 history
#   7. blanks demerger ex-dates (demergers.csv), ex-rights falls (rights_events.csv) and returns
#      spanning feed-gap days (weekdays where many stocks are missing)
#   8. sets any remaining |return| > 50% to NaN (unlisted demergers, bad ticks) -- the row is kept
#      for price charts, but the model and backtest skip that day
# Before ANGEL_START the stored returns are kept (Yahoo 2000-2019 is already split- and dividend-
# adjusted; no dividends are added to the Jan 2020 - Apr 2021 gap fill): only steps 1 and 8 apply,
# and returns across gaps of more than 10 days or at adjusted prices below Rs 1 are blanked.

import numpy as np
import pandas as pd

from corporate_actions import load_actions, load_demergers, load_dividends, load_rights

ANGEL_START   = pd.Timestamp("2021-04-12")
MAX_GAP_DAYS  = 10            # recompute returns only across normal gaps (weekends, holidays)
BAND          = (-0.20, 0.25) # NSE circuit bands cap normal daily moves at ~20%
MAX_ABS_RET   = 0.50
SPARSE_FRAC   = 0.50          # a date with fewer stocks than this share of the usual count is dropped
MIN_PRICE     = 1.0           # before ANGEL_START, returns at adjusted prices below this are blanked
RIGHTS_WINDOW_DAYS = 15       # an ex-rights date falls within this many days before the entitlements trade
RIGHTS_MIN_DROP = -0.10       # ... and shows as the stock's largest fall vs the market in that window, below this
RIGHTS_CALM_MARKET = -0.03    # ... on a day the median stock fell less than this (not a market-wide crash)
BIG_DIVIDEND  = 0.03          # dividend yields above this must show up as a price drop to be used
ACTION_WINDOW_DAYS = 3        # a recorded split/bonus may be dated a day or two from the price move
OFF_DATE_TOL  = 0.03          # off the recorded ex-date, the adjusted move must be smaller than this
COMBINE_FACTORS = [2, 1.5, 1.25, 3, 4, 5, 10, 0.5]   # extra factor when a split and a bonus share a day
# price multipliers of splits/bonuses (as ratio new/old shares) used only to flag unrecorded ones
SUSPECT_FACTORS = [2, 3, 1.5, 4, 5, 10, 20, 1 / 2, 1 / 3, 1 / 5, 1 / 10]


def sparse_dates(df: pd.DataFrame, frac: float = SPARSE_FRAC) -> pd.DatetimeIndex:
    """Dates (from ANGEL_START) with fewer than `frac` of the usual (rolling 21-day median) stock count."""
    cnt = df[df["date"] >= ANGEL_START].groupby("date")["stock"].size()
    med = cnt.rolling(21, center=True, min_periods=5).median()
    return cnt.index[cnt < frac * med]


def clean(df: pd.DataFrame, verbose: bool = True, actions: pd.DataFrame = None,
          dividends: pd.DataFrame = None, demergers: pd.DataFrame = None, rights: pd.DataFrame = None) -> pd.DataFrame:
    n0 = len(df)
    df = df[~df["stock"].str.contains("NSETEST", na=False)]
    angel = df["date"] >= ANGEL_START
    # no-trade rows: from ANGEL_START all of them; before, the stale ones (zero volume and an exactly
    # unchanged price: the source carried the last price forward; the next row's return still spans it)
    stale = (df["volume"] <= 0) & (angel | (df["return_1d"] == 0))
    df = df[~stale]
    n_dropped = n0 - len(df)
    sparse = sparse_dates(df)
    df = df[~df["date"].isin(sparse)]

    # OHLC consistency: the day's high/low must contain the open and close
    px = df[["open", "high", "low", "close"]]
    hi, lo = px.max(axis=1), px.min(axis=1)
    n_ohlc = int(((df["high"] < hi) | (df["low"] > lo)).sum())
    df = df.assign(high=hi, low=lo)

    df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    g = df.groupby("stock", sort=False)
    prev_close = g["close"].shift()
    gap = (df["date"] - g["date"].shift()).dt.days

    ret = df["return_1d"].astype(float).copy()
    recompute = (df["date"] >= ANGEL_START) & (gap <= MAX_GAP_DAYS) & (prev_close > 0)
    ret[recompute] = df.loc[recompute, "close"] / prev_close[recompute] - 1
    # From ANGEL_START, a return across a gap of more than MAX_GAP_DAYS (suspension) is not a 1-day
    # return: blank it for the model (the backtest still books the real move).
    # Before ANGEL_START the stored returns are kept, except across such gaps and at tiny adjusted
    # prices (old Yahoo history scaled down by many bonuses, where rounding makes returns garbage)
    ret[(df["date"] >= ANGEL_START) & ~recompute] = np.nan
    early = df["date"] < ANGEL_START
    ret[early & ((gap > MAX_GAP_DAYS) | (df["close"] < MIN_PRICE) | (prev_close < MIN_PRICE))] = np.nan

    # Splits / bonus issues: adjusted ONLY where a split/bonus is recorded for this stock (from
    # Yahoo, or an ISIN change in the bhavcopies), on one day per event, and only if the price
    # actually moved by at least half the split factor. Guessing from the size of the drop alone is
    # unsafe: a -20% lower circuit or a -28% crash looks like a 1:2 bonus (price x 2/3).
    actions = load_actions() if actions is None else actions
    ratio = (df["close"] / prev_close).values
    dates, rec = df["date"].values, recompute.values
    adjusted = np.zeros(len(df), bool)
    n_split = 0
    if len(actions):
        rows = g.indices
        win = np.timedelta64(ACTION_WINDOW_DAYS, "D")
        for s, d, k in actions[["stock", "date", "ratio"]].itertuples(index=False):
            idx = rows.get(s)
            if idx is None or not k > 0 or k == 1:
                continue
            d = np.datetime64(pd.Timestamp(d))
            near = idx[(np.abs(dates[idx] - d) <= win) & rec[idx] & ~adjusted[idx]]
            if not len(near):
                continue
            r = ratio[near]
            adj = r * k - 1
            moved = r < 1 - 0.5 * (1 - 1 / k) if k > 1 else r > 1 + 0.5 * (1 / k - 1)
            ok = moved & (np.abs(adj) < np.abs(ret.values[near]))   # the recorded action explains the move
            if not ok.any():
                continue
            # the ex-date itself if it qualifies, else a nearby day only if the adjustment leaves almost
            # no move (so a real drop near the ex-date of a small bonus is never taken for it)
            exact = ok & (dates[near] == d)
            ok_near = ok & (np.abs(adj) < OFF_DATE_TOL)
            if exact.any():
                j = int(np.flatnonzero(exact)[0])
            elif ok_near.any():
                j = int(np.nanargmin(np.where(ok_near, np.abs(adj), np.inf)))
            else:
                continue
            i = near[j]
            a = adj[j]
            if not BAND[0] <= a <= BAND[1]:
                # still far outside a normal day: a split and a bonus on the same day where only one was
                # recorded (e.g. 1:5 split + 1:1 bonus = 10x). Try the combined factor; else blank the day
                # several combinations often fit, so use one only when it is the only fit or leaves
                # almost no move; otherwise blank the day rather than guess
                combos = [ratio[i] * k * m - 1 for m in COMBINE_FACTORS]
                inside = [c for c in combos if BAND[0] <= c <= BAND[1]]
                tiny = [c for c in inside if abs(c) < OFF_DATE_TOL]
                a = inside[0] if len(inside) == 1 else (min(tiny, key=abs) if tiny else np.nan)
            ret.iat[i] = a
            adjusted[i] = True
            n_split += 1

    # A big move whose price ratio is exactly a split factor (within 1%) but has no recorded
    # action is probably an unrecorded split/bonus: blank it rather than trust it. (-20% circuit
    # days, ratio 0.8, are deliberately not in this list -- they are far more often real.)
    still_out = recompute & ~pd.Series(adjusted) & ((ret < BAND[0]) | (ret > BAND[1]))
    rr = pd.Series(ratio)[still_out]
    suspect_idx = rr.index[np.min([np.abs(rr * k - 1) for k in SUSPECT_FACTORS], axis=0) < 0.01] if len(rr) else []
    ret[suspect_idx] = np.nan
    n_suspect = len(suspect_idx)

    # Cash dividends on the ex-date (yield = dividend / previous close), not on split days
    dividends = load_dividends() if dividends is None else dividends
    n_div = 0
    if len(dividends):
        key = pd.MultiIndex.from_frame(df[["stock", "date"]])
        dv = dividends.assign(date=pd.to_datetime(dividends["date"])).drop_duplicates(["stock", "date"])
        y = pd.Series(dv["yield"].values, index=pd.MultiIndex.from_frame(dv[["stock", "date"]])).reindex(key).values
        ok = ~np.isnan(y) & rec & ~adjusted & ret.notna().values
        # A large dividend must show in the price (it falls by at least half of it on the ex-date);
        # otherwise the record is wrong or the feed already adjusted for it, and it is skipped
        with np.errstate(invalid="ignore"):
            ok &= (y <= BIG_DIVIDEND) | (ratio - 1 < -0.5 * y)
        ret[ok] = ret[ok] + y[ok]
        n_div = int(ok.sum())

    # Demergers (demergers.csv): on the ex-date the price falls by the value of the business spun
    # off, which shareholders receive as new shares. That is not a loss, so the return is blanked
    # (only if the data shows a fall that day, as a check on the date)
    demergers = load_demergers() if demergers is None else demergers
    n_dem = 0
    if len(demergers):
        dm = pd.MultiIndex.from_frame(demergers.assign(date=pd.to_datetime(demergers["date"]))[["stock", "date"]])
        hit = pd.MultiIndex.from_frame(df[["stock", "date"]]).isin(dm) & (ret.values < -0.03)
        ret[hit] = np.nan
        n_dem = int(hit.sum())

    # Rights issues (rights_events.csv: stock, date the rights entitlements started trading): the
    # ex-rights fall is a dilution shareholders are compensated for by the entitlements, not a loss.
    # In the RIGHTS_WINDOW_DAYS before, the stock's largest fall RELATIVE TO THE MARKET (median stock
    # that day) is blanked if it is below RIGHTS_MIN_DROP; a market-wide crash day is never taken for
    # it. Once per event, from ANGEL_START only (returns there are recomputed, so this is repeatable).
    rights = load_rights() if rights is None else rights
    n_rights = 0
    if len(rights):
        rows = g.indices
        late = (df["date"] >= ANGEL_START).values
        market = ret[late].groupby(df.loc[late, "date"]).median()
        mkt = df["date"].map(market).values
        rel = ret.values - mkt
        rel[mkt < RIGHTS_CALM_MARKET] = np.nan          # never on a market-wide sell-off day
        done = set()
        for st_, d in rights[["stock", "date"]].drop_duplicates().itertuples(index=False):
            idx = rows.get(st_)
            if idx is None:
                continue
            d = np.datetime64(pd.Timestamp(d))
            w = idx[(dates[idx] < d) & (dates[idx] >= d - np.timedelta64(RIGHTS_WINDOW_DAYS, "D")) & late[idx]]
            w = np.array([i for i in w if i not in done], dtype=int)
            if len(w) and np.nanmin(rel[w], initial=0) < RIGHTS_MIN_DROP:
                i = w[int(np.nanargmin(rel[w]))]
                ret.iat[i] = np.nan
                done.add(i)
                n_rights += 1

    # Feed gaps: a weekday where many stocks are missing even though they trade the days
    # around it (neither source has the day). The next return for those stocks would span
    # 2 days, so it is blanked instead of being mislabelled as a 1-day return.
    cnt = df[df["date"] >= ANGEL_START].groupby("date")["stock"].size()
    med = cnt.rolling(21, center=True, min_periods=5).median()
    gap_days = cnt.index[(cnt < 0.85 * med) & (cnt.index.dayofweek < 5)]
    prev_date = g["date"].shift()
    n_gap = 0
    for d in gap_days:
        spans = recompute & (prev_date < d) & (df["date"] > d)
        n_gap += int(spans.sum())
        ret[spans] = np.nan

    bad = ret.abs() > MAX_ABS_RET
    bad |= df["close"] <= 0
    ret[bad] = np.nan
    df["return_1d"] = ret.round(6)

    if verbose:
        print(f"  clean: fixed {n_ohlc:,} high/low ranges, "
              f"dropped {n_dropped:,} rows (test symbols / no-trade days) and {len(sparse)} sparse dates "
              f"({', '.join(str(d.date()) for d in sparse) or 'none'}), "
              f"recomputed {int(recompute.sum()):,} returns, adjusted {n_split} recorded splits/bonuses, "
              f"blanked {n_suspect} suspected unrecorded ones, added {n_div:,} dividends, blanked {n_dem} demerger and {n_rights} ex-rights days, "
              f"blanked {n_gap:,} returns spanning feed-gap days "
              f"({', '.join(str(d.date()) for d in gap_days) or 'none'}) and {int(bad.sum()):,} implausible returns")
    return df.sort_values(["date", "stock"]).reset_index(drop=True)


if __name__ == "__main__":
    import os
    data = pd.read_csv("dataset.csv", parse_dates=["date"])
    data = clean(data)
    data.to_csv("dataset.csv.tmp", index=False)
    os.replace("dataset.csv.tmp", "dataset.csv")
    print(f"Saved dataset.csv ({len(data):,} rows)")
