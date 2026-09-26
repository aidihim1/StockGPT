# audit_data.py -- data-quality report for dataset.csv
#
# Checks structure, coverage, trading calendar, missing days, return consistency,
# extreme moves, OHLC sanity, and cross-checks recent closes against Yahoo Finance.
#
# Usage:  python audit_data.py            (add --no-web to skip the Yahoo cross-check)

import argparse
import numpy as np
import pandas as pd

ANGEL_START = pd.Timestamp("2021-04-12")
# NSE special sessions that legitimately fall on weekends/holidays (Muhurat, DR drills, Budget day)
SPECIAL_SESSIONS = {"2023-11-12", "2024-01-20", "2024-03-02", "2024-05-18", "2024-11-01",
                    "2025-02-01", "2025-10-21", "2026-02-01"}


def main(args):
    df = pd.read_csv("dataset.csv", parse_dates=["date"]).sort_values(["stock", "date"])
    sym = set(pd.read_csv("nse_symbols.csv")["clean_symbol"])
    last_day = df["date"].max()
    ok = lambda b: "OK " if b else "!! "

    print("=" * 78)
    print("1. STRUCTURE")
    print(f"   rows {len(df):,} | stocks {df.stock.nunique():,} | {df.date.min().date()} -> {last_day.date()}")
    dups = df.duplicated(["date", "stock"]).sum()
    print(f"   {ok(dups == 0)}duplicate (date, stock) rows: {dups}")
    nan_px = df[["open", "high", "low", "close", "volume"]].isna().sum().sum()
    print(f"   {ok(nan_px == 0)}missing price/volume values: {nan_px}")
    print(f"   -- returns left blank on purpose (splits/demergers/bad ticks): {df.return_1d.isna().sum():,}")

    print("\n2. COVERAGE")
    last = df.groupby("stock")["date"].max()
    listed_in_data = last[last.index.isin(sym)]
    stale = listed_in_data[listed_in_data < last_day]
    print(f"   listed stocks up to date ({last_day.date()}): {(listed_in_data == last_day).sum():,} of {len(listed_in_data):,}")
    print(f"   {ok(len(stale) <= 20)}listed stocks behind: {len(stale)}"
          + (f"  -> {dict(list(stale.dt.date.astype(str).items())[:10])}" if len(stale) else ""))
    missing = sorted(sym - set(df.stock))
    print(f"   listed symbols with no data at all: {len(missing)} {missing[:10]}")
    print(f"   delisted/renamed stocks kept for history: {(~last.index.isin(sym)).sum()}")

    print("\n3. TRADING CALENDAR (since Apr 2021)")
    a = df[df.date >= ANGEL_START]
    cnt = a.groupby("date")["stock"].nunique()
    wk = cnt[cnt.index.dayofweek >= 5]
    odd_wk = [d.date() for d in wk.index if str(d.date()) not in SPECIAL_SESSIONS]
    print(f"   {ok(not odd_wk)}weekend dates with data (excluding known special sessions): {odd_wk}")
    med = cnt.rolling(21, center=True, min_periods=5).median()
    thin = cnt[(cnt < 0.8 * med) & ~cnt.index.isin(wk.index)]
    print(f"   {ok(len(thin) <= 3)}weekdays with unusually few stocks: "
          f"{ {str(k.date()): int(v) for k, v in thin.items()} }")
    print(f"   -- known gap: no data 2020-01-01 -> 2021-04-12 (between Yahoo and Angel One history)")

    print("\n4. MISSING DAYS PER STOCK (since Apr 2021, between each stock's first and last date)")
    mkt = cnt[cnt >= 0.5 * med].index
    g = a.groupby("stock")["date"]
    first, lastd, n = g.min(), g.max(), g.count()
    pos = lambda d: np.searchsorted(mkt.values, d.values)
    expected = pos(lastd) - pos(first) + 1
    miss = pd.Series(expected - n.values, index=n.index).clip(lower=0)
    listed_miss = miss[miss.index.isin(sym)]
    print(f"   listed stocks: 0 missing {int((listed_miss == 0).sum())} | 1-5 {int(listed_miss.between(1, 5).sum())} | "
          f"6-20 {int(listed_miss.between(6, 20).sum())} | >20 {int((listed_miss > 20).sum())}")
    print(f"   (missing days are usually genuine no-trade days in illiquid stocks or suspensions)")
    print(f"   most gaps: {listed_miss.sort_values(ascending=False).head(6).to_dict()}")

    print("\n5. RETURN CONSISTENCY (Angel era)")
    prev = df.groupby("stock")["close"].shift()
    gap = (df["date"] - df.groupby("stock")["date"].shift()).dt.days
    m = (df.date >= ANGEL_START) & (gap <= 10) & df.return_1d.notna() & (prev > 0)
    diff = ((df.close / prev - 1) - df.return_1d).abs()[m]
    n_bad = int((diff > 0.005).sum())
    print(f"   rows where return_1d != close/prev_close - 1: {n_bad:,} of {int(m.sum()):,} "
          f"(expected: split/bonus days, which are adjusted on purpose)")
    r = df.loc[df.date >= ANGEL_START, "return_1d"]
    print(f"   {ok((r.abs() > 0.5).sum() == 0)}|return| > 50%: {(r.abs() > 0.5).sum()} | > 20%: {(r.abs() > 0.2).sum()} "
          f"(NSE circuit bands cap most moves at 20%)")

    print("\n6. PRICE SANITY (Angel era)")
    p = df[df.date >= ANGEL_START]
    checks = {
        "price <= 0": (p[["open", "high", "low", "close"]] <= 0).any(axis=1).sum(),
        "high < low": (p.high < p.low).sum(),
        "close outside [low, high]": ((p.close > p.high * 1.001) | (p.close < p.low * 0.999)).sum(),
        "open outside [low, high]": ((p.open > p.high * 1.001) | (p.open < p.low * 0.999)).sum(),
        "volume 0 (no-trade filler)": (p.volume <= 0).sum(),
    }
    for k, v in checks.items():
        print(f"   {ok(v == 0)}{k}: {v}")

    if not args.no_web:
        print("\n7. CROSS-CHECK LAST 4 DAYS vs YAHOO FINANCE (raw closes)")
        import yfinance as yf
        big = (p[p.date >= last_day - pd.Timedelta(days=40)].assign(tv=lambda x: x.close * x.volume)
               .groupby("stock")["tv"].mean().sort_values(ascending=False))
        rng = np.random.default_rng(0)
        sample = list(big.index[:15]) + list(rng.choice(big.index[15:], 15, replace=False))
        raw = yf.download([s + ".NS" for s in sample], start=str((last_day - pd.Timedelta(days=7)).date()),
                          auto_adjust=False, progress=False, group_by="ticker")
        rows = []
        for s in sample:
            try:
                y = raw[s + ".NS"]["Close"].dropna()
                y.index = pd.to_datetime(y.index).tz_localize(None).normalize()
            except Exception:
                continue
            ours = df[(df.stock == s) & (df.date >= y.index.min())].set_index("date")["close"]
            both = pd.concat([ours, y], axis=1, keys=["ours", "yahoo"]).dropna()
            if len(both):
                rows.append((s, len(both), float((both.ours / both.yahoo - 1).abs().max() * 100)))
        res = pd.DataFrame(rows, columns=["stock", "days", "max_diff_pct"])
        bad = res[res.max_diff_pct > 0.5]
        print(f"   compared {len(res)} stocks (15 most traded + 15 random), {int(res.days.sum())} closes")
        print(f"   {ok(bad.empty)}closes differing by more than 0.5%: {len(bad)}"
              + (f"\n{bad.to_string(index=False)}" if len(bad) else ""))
        print(f"   median difference: {res.max_diff_pct.median():.3f}%")

    print("=" * 78)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-web", action="store_true")
    main(p.parse_args())
