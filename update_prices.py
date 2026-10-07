# update_prices.py -- the data update (run before new picks; once a month is enough)
#
# Uses NSE's official daily bhavcopy files, so every price comes from one consistent source:
#   1. fetch_bhavcopy.py          downloads the days not yet in bhavcopy/
#   2. etf_list.py                refreshes the ETF list and each company's NSE series (EQ / BE / BZ)
#   3. corporate_actions.py       refreshes recorded splits/bonuses and dividends (last 400 days)
#   4. rebuild_from_bhavcopy.py   rebuilds prices from 12 Apr 2021 on and cleans them
# Every calendar day up to today is tried (weekends too, for special sessions); NSE publishes no file
# for holidays, and days without a file are remembered and not asked for again once a week old.
# update_data.py (Angel One API) is kept as a fallback if NSE's archive is unavailable.
#
# Usage:  python update_prices.py

import subprocess
import sys

import pandas as pd

STEPS = [
    ["fetch_bhavcopy.py", "--start", "2021-04-12", "--calendar"],
    ["etf_list.py"],
    ["corporate_actions.py", "--all", "--days", "400"],
    ["rebuild_from_bhavcopy.py"],
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n=== {' '.join(step)} ===", flush=True)
        r = subprocess.run([sys.executable, "-u"] + step)
        if r.returncode != 0:
            raise SystemExit(f"{step[0]} failed (exit {r.returncode}); data left as it was before that step.")
    print(f"\nData updated ({pd.Timestamp.now():%d %b %Y %H:%M}). Now run picks.py for fresh picks.")
