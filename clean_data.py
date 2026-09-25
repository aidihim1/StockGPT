# clean_data.py -- consistency fixes for dataset.csv
#
# update_data.py runs clean() after every update. Can also be run on its own:
#   python clean_data.py
#
# From ANGEL_START on, prices are raw exchange closes (Angel One; yfinance fallback
# with auto_adjust=False). clean():
#   1. drops dummy test symbols (e.g. 011NSETEST)
#   2. drops no-trade rows (volume 0) from ANGEL_START on -- data vendors emit these on
#      NSE holidays and for untraded stocks, which would feed fake flat days to the model
#   3. recomputes return_1d = close / previous close - 1, so a day missing from one source
#      can never turn a 2-day move into a "1-day" return (rows after a >10-day gap keep
#      the return computed at fetch time)
#   4. neutralises splits / bonus issues: a move beyond the circuit band whose price ratio
#      matches a split factor (e.g. x0.1 on a 1:10 split) is replaced by the split-adjusted
#      return
#   5. sets any remaining |return| > 50% to NaN (demergers, bad ticks) -- the row is kept
#      for price charts, but the model and backtest skip that day
# Pre-2021 Yahoo data is already split-adjusted, so only steps 1 and 5 apply to it.

import numpy as np
import pandas as pd

ANGEL_START   = pd.Timestamp("2021-04-12")
MAX_GAP_DAYS  = 10            # recompute returns only across normal gaps (weekends, holidays)
BAND          = (-0.20, 0.25) # NSE circuit bands cap normal daily moves at ~20%
MAX_ABS_RET   = 0.50
# Price multipliers of common corporate actions: splits 1:2, 1:4, 1:5, 1:10, 1:20,
# bonus 1:1 (x0.5), 1:2 (x2/3), 2:1 (x1/3), 3:1 (x0.25), and reverse splits
SPLIT_FACTORS = [2, 3, 4, 5, 10, 20, 1.5, 1 / 2, 1 / 5, 1 / 10]


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    n0 = len(df)
    df = df[~df["stock"].str.contains("NSETEST", na=False)]
    angel = df["date"] >= ANGEL_START
    df = df[~(angel & (df["volume"] <= 0))]
    n_dropped = n0 - len(df)

    df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    g = df.groupby("stock", sort=False)
    prev_close = g["close"].shift()
    gap = (df["date"] - g["date"].shift()).dt.days

    ret = df["return_1d"].astype(float).copy()
    recompute = (df["date"] >= ANGEL_START) & (gap <= MAX_GAP_DAYS) & (prev_close > 0)
    ret[recompute] = df.loc[recompute, "close"] / prev_close[recompute] - 1

    # Splits / bonus issues
    ratio = df["close"] / prev_close
    out_of_band = recompute & ((ret < BAND[0]) | (ret > BAND[1]))
    n_split = 0
    for i in np.flatnonzero(out_of_band.values):
        r = ratio.iat[i]
        best = min(SPLIT_FACTORS, key=lambda k: abs(r * k - 1))
        if abs(r * best - 1) < 0.20:
            ret.iat[i] = r * best - 1
            n_split += 1

    bad = ret.abs() > MAX_ABS_RET
    bad |= df["close"] <= 0
    ret[bad] = np.nan
    df["return_1d"] = ret.round(6)

    if verbose:
        print(f"  clean: dropped {n_dropped:,} rows (test symbols / no-trade days), "
              f"recomputed {int(recompute.sum()):,} returns, adjusted {n_split} splits/bonuses, "
              f"blanked {int(bad.sum()):,} implausible returns")
    return df.sort_values(["date", "stock"]).reset_index(drop=True)


if __name__ == "__main__":
    data = pd.read_csv("dataset.csv", parse_dates=["date"])
    data = clean(data)
    data.to_csv("dataset.csv", index=False)
    print(f"Saved dataset.csv ({len(data):,} rows)")
