# clean_data.py -- consistency fixes for dataset.csv
#
# update_data.py runs clean() after every update. Can also be run on its own:
#   python clean_data.py
#
# From ANGEL_START on, prices are raw exchange closes (Angel One; yfinance fallback
# with auto_adjust=False). clean():
#   1. drops dummy test symbols (e.g. 011NSETEST)
#   2. drops no-trade rows (volume 0) from ANGEL_START on -- data vendors emit these on
#      NSE holidays and for untraded stocks, which would feed fake flat days to the model;
#      widens high/low where the open or close falls outside them
#   3. recomputes return_1d = close / previous close - 1, so a day missing from one source
#      can never turn a 2-day move into a "1-day" return (rows after a >10-day gap keep
#      the return computed at fetch time)
#   4. neutralises splits / bonus issues: a move beyond the circuit band is replaced by the
#      split-adjusted return ONLY if corporate_actions.csv records a split/bonus for that stock
#      within ACTION_WINDOW_DAYS (never guessed from the size of the move)
#   5. sets any remaining |return| > 50% to NaN (demergers, bad ticks) -- the row is kept
#      for price charts, but the model and backtest skip that day
# Pre-2021 Yahoo data is already split-adjusted, so only steps 1 and 5 apply to it.

import numpy as np
import pandas as pd

from corporate_actions import load_actions

ANGEL_START   = pd.Timestamp("2021-04-12")
MAX_GAP_DAYS  = 10            # recompute returns only across normal gaps (weekends, holidays)
BAND          = (-0.20, 0.25) # NSE circuit bands cap normal daily moves at ~20%
MAX_ABS_RET   = 0.50
ACTION_WINDOW_DAYS = 3        # a recorded split/bonus may be dated a day or two from the price move
# price multipliers of splits/bonuses (as ratio new/old shares) used only to flag unrecorded ones
SUSPECT_FACTORS = [2, 3, 1.5, 4, 5, 10, 20, 1 / 2, 1 / 3, 1 / 5, 1 / 10]


def clean(df: pd.DataFrame, verbose: bool = True, actions: pd.DataFrame = None) -> pd.DataFrame:
    n0 = len(df)
    df = df[~df["stock"].str.contains("NSETEST", na=False)]
    angel = df["date"] >= ANGEL_START
    df = df[~(angel & (df["volume"] <= 0))]
    n_dropped = n0 - len(df)

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

    # Splits / bonus issues: adjusted ONLY where a split/bonus is recorded for this stock within
    # a few days (corporate_actions.csv, from Yahoo). Guessing from the size of the drop is
    # unsafe: a -20% lower circuit or a -28% crash looks like a 1:2 bonus (price x 2/3).
    actions = load_actions() if actions is None else actions
    ratio = df["close"] / prev_close
    n_split = 0
    if len(actions):
        events = {s: e for s, e in actions.assign(date=pd.to_datetime(actions["date"])).groupby("stock")}
        cand = recompute & ((ret < BAND[0]) | (ret > BAND[1])) & df["stock"].isin(events.keys())
        for i in np.flatnonzero(cand.values):
            e, d = events[df["stock"].iat[i]], df["date"].iat[i]
            near = e[(e["date"] - d).abs() <= pd.Timedelta(days=ACTION_WINDOW_DAYS)]
            if near.empty:
                continue
            k = float(near["ratio"].iloc[(near["date"] - d).abs().argmin()])
            adj = ratio.iat[i] * k - 1
            if abs(adj) < abs(ret.iat[i]):           # the recorded action explains the move
                ret.iat[i] = adj
                n_split += 1

    # A big move whose price ratio is exactly a split factor (within 1%) but has no recorded
    # action is probably an unrecorded split/bonus: blank it rather than trust it. (-20% circuit
    # days, ratio 0.8, are deliberately not in this list -- they are far more often real.)
    still_out = recompute & ((ret < BAND[0]) | (ret > BAND[1]))
    rr = ratio[still_out]
    suspect_idx = rr.index[np.min([np.abs(rr * k - 1) for k in SUSPECT_FACTORS], axis=0) < 0.01] if len(rr) else []
    ret[suspect_idx] = np.nan
    n_suspect = len(suspect_idx)

    # Feed gaps: a weekday where many stocks are missing even though they trade the days
    # around it (neither Angel One nor Yahoo has the day). The next return for those stocks
    # would span 2 days, so it is blanked instead of being mislabelled as a 1-day return.
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
              f"dropped {n_dropped:,} rows (test symbols / no-trade days), "
              f"recomputed {int(recompute.sum()):,} returns, adjusted {n_split} recorded splits/bonuses, "
              f"blanked {n_suspect} suspected unrecorded ones, "
              f"blanked {n_gap:,} returns spanning feed-gap days "
              f"({', '.join(str(d.date()) for d in gap_days) or 'none'}) and {int(bad.sum()):,} implausible returns")
    return df.sort_values(["date", "stock"]).reset_index(drop=True)


if __name__ == "__main__":
    data = pd.read_csv("dataset.csv", parse_dates=["date"])
    data = clean(data)
    data.to_csv("dataset.csv", index=False)
    print(f"Saved dataset.csv ({len(data):,} rows)")
