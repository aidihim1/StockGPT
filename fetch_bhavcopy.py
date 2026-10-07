# fetch_bhavcopy.py -- downloads NSE's official daily equity "bhavcopy" files
#
# A bhavcopy lists every security traded on NSE that day (symbol, series, ISIN, OHLC, volume, value).
# Unlike a broker's symbol list, it includes companies that were later delisted or merged, so
# rebuild_from_bhavcopy.py builds every price from 12 Apr 2021 on from these files (no survivorship
# bias from then on). Raw (unadjusted) prices. One zip per trading day in bhavcopy/; days already
# downloaded are skipped, and days with no file (holidays) are remembered in bhavcopy/no_file_days.txt.
#
# Usage:  python fetch_bhavcopy.py --start 2021-04-12 --calendar   (every day up to today, as update_prices.py)
#         python fetch_bhavcopy.py --start 2021-04-12              (only the dates already in dataset.csv)
#         options: --end YYYY-MM-DD, --pause SECONDS between downloads (default 0.4)

import argparse
import os
import time

import pandas as pd
import requests

from etf_list import HEADERS

OUT = "bhavcopy"
UDIFF_FROM = pd.Timestamp("2024-07-08")      # NSE switched to the UDiFF file format on this day


def urls(d: pd.Timestamp) -> list:
    old = (f"https://nsearchives.nseindia.com/content/historical/EQUITIES/{d.year}/{d.strftime('%b').upper()}/"
           f"cm{d.strftime('%d')}{d.strftime('%b').upper()}{d.year}bhav.csv.zip")
    new = f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    return [new, old] if d >= UDIFF_FROM else [old, new]


def path_for(d: pd.Timestamp) -> str:
    return os.path.join(OUT, f"{d:%Y%m%d}.csv.zip")


def fetch_day(session, d: pd.Timestamp, retries: int = 3) -> str:
    """'ok' (saved), 'none' (NSE answered 'not found' for both file formats: no session that day) or
    'error' (network trouble, blocked, or an unexpected reply: try again on a later run)."""
    for attempt in range(retries):
        statuses = []
        for u in urls(d):
            try:
                r = session.get(u, headers=HEADERS, timeout=30)
            except requests.RequestException:
                statuses.append(None)
                continue
            statuses.append(r.status_code)
            if r.status_code == 200 and r.content[:2] == b"PK":          # a zip file
                tmp = path_for(d) + ".tmp"
                with open(tmp, "wb") as fh:
                    fh.write(r.content)
                os.replace(tmp, path_for(d))
                return "ok"
        if all(st == 404 for st in statuses):
            return "none"
        time.sleep(2 * (attempt + 1))
    return "error"


NO_FILE = os.path.join(OUT, "no_file_days.txt")     # holidays: days NSE published no bhavcopy for


def main(args):
    os.makedirs(OUT, exist_ok=True)
    if args.calendar:          # every day up to today, weekends too (special sessions); holidays are remembered
        days = pd.date_range(args.start, args.end or pd.Timestamp.now().normalize())
    else:
        days = pd.read_csv("dataset.csv", usecols=["date"], parse_dates=["date"])["date"].unique()
        days = pd.DatetimeIndex(sorted(days))
        days = days[(days >= pd.Timestamp(args.start)) & (days <= pd.Timestamp(args.end or days.max()))]
    known_none = set(open(NO_FILE).read().split()) if os.path.exists(NO_FILE) else set()
    recent = pd.Timestamp.now().normalize() - pd.Timedelta(days=7)
    todo = [d for d in days if not os.path.exists(path_for(d)) and (f"{d:%Y%m%d}" not in known_none or d >= recent)]
    print(f"{len(days)} trading days, {len(todo)} to download", flush=True)
    session, none, failed, errors = requests.Session(), [], [], []
    settled = pd.Timestamp.now().normalize() - pd.Timedelta(days=3)    # today's file may not be out yet
    for i, d in enumerate(todo, 1):
        status = fetch_day(session, d)
        if status == "none" and d <= settled:
            none.append(d)
        elif status != "ok":
            failed.append(d.date())
            if status == "error":
                errors.append(d.date())
        time.sleep(args.pause)
        if i % 100 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} | no session {len(none)} | failed {len(failed)}", flush=True)
    if none:
        with open(NO_FILE, "a") as fh:
            fh.writelines(f"{d:%Y%m%d}" + os.linesep for d in none)
    failed += [d.date() for d in none]
    print(f"Done. {len(days) - len(failed)} of {len(days)} days available in {OUT}/")
    if failed:
        print(f"  no file for {len(failed)} days (holidays/special sessions or not published): {failed[:20]}")
    if errors:
        # A missing trading day would silently vanish from every stock's history, so stop the update
        raise SystemExit(f"Download failed for {len(errors)} days (network or NSE trouble): {errors[:20]}. "
                         f"Run again later.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2021-04-12")
    p.add_argument("--end", default=None)
    p.add_argument("--pause", type=float, default=0.4)
    p.add_argument("--calendar", action="store_true", help="every day up to today instead of dataset dates")
    main(p.parse_args())
