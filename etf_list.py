# etf_list.py -- which symbols are ETFs (index, gold, silver, liquid/money-market funds), not companies
#
# The price data covers everything Angel One lists as NSE "-EQ", which includes ~360 ETFs. They are
# kept in dataset.csv (NIFTYBEES is the Nifty benchmark) but excluded from the stock universe:
# a liquid fund is cash, not a stock.
#
# etf_symbols.csv columns: stock, source
#   source = "NSE ETF list" (NSE's official list, ISINs starting INF) or "renamed/matured ETF"
#   (ETFs in our history that are no longer on NSE's list). Entries are never removed, so ETFs
#   that were later renamed or delisted stay excluded.
#
# Usage:  python etf_list.py      # refresh from NSE (update_data.py also does this)

import os

import pandas as pd

FILE = "etf_symbols.csv"
NSE_ETF_URL = "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/126.0 Safari/537.36", "Referer": "https://www.nseindia.com/"}


def load_etfs() -> set:
    if not os.path.exists(FILE):
        raise SystemExit(f"{FILE} not found. Run `python etf_list.py` first.")
    return set(pd.read_csv(FILE)["stock"])


def refresh_etfs() -> int:
    """Add any ETFs on NSE's current list to etf_symbols.csv. Returns the number added."""
    import io
    import requests
    resp = requests.get(NSE_ETF_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    nse = pd.read_csv(io.StringIO(resp.text))
    nse.columns = nse.columns.str.strip()
    nse = nse[nse["ISINNumber"].str.strip().str.startswith("INF")]
    old = pd.read_csv(FILE) if os.path.exists(FILE) else pd.DataFrame(columns=["stock", "source"])
    new = pd.DataFrame({"stock": nse["Symbol"].str.strip(), "source": "NSE ETF list"})
    new = new[~new["stock"].isin(old["stock"])]
    if len(new):
        pd.concat([old, new]).sort_values("stock").to_csv(FILE, index=False)
    return len(new)


if __name__ == "__main__":
    n = refresh_etfs()
    print(f"{FILE}: {len(load_etfs())} ETFs ({n} new)")
