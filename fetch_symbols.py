# fetch_symbols.py  —  downloads the master list of all tradable symbols

import requests
import pandas as pd
import json

def download_instrument_master():
    """
    Angel One provides a JSON file with every tradable instrument.
    We filter it to get only active NSE equity stocks.
    """
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    
    print("Downloading instrument master file... (this is ~20MB, takes a moment)")
    response = requests.get(url, timeout=60)
    data = response.json()
    
    df = pd.DataFrame(data)
    
    # Filter: only NSE exchange, only cash equities (not futures/options)
    # 'expiry' is empty for equity stocks (derivatives have an expiry date)
    # Series: EQ (normal), BE/BZ (trade-for-trade: surveillance or non-compliant companies). BE/BZ
    # must be kept: leaving out stocks in trouble would flatter backtests. ETFs are excluded later
    # (etf_list.py).
    nse_stocks = df[
        (df["exch_seg"] == "NSE") &                        # NSE exchange only
        (df["instrumenttype"] == "")  &                     # equity (not F&O)
        (df["symbol"].str.contains(r"-(?:EQ|BE|BZ)$"))      # cash equity series
    ].copy()

    # Clean symbol name (remove the series suffix); one row per company, EQ token preferred
    nse_stocks["clean_symbol"] = nse_stocks["symbol"].str.replace(r"-(?:EQ|BE|BZ)$", "", regex=True)
    nse_stocks["order"] = nse_stocks["symbol"].str[-2:].map({"EQ": 0, "BE": 1, "BZ": 2})
    nse_stocks = nse_stocks.sort_values("order").drop_duplicates("clean_symbol")

    # Keep only the columns we need
    nse_stocks = nse_stocks[["token", "symbol", "name", "exch_seg", "clean_symbol"]].reset_index(drop=True)
    
    nse_stocks.to_csv("nse_symbols.csv", index=False)
    print(f"Saved {len(nse_stocks)} NSE equity stocks to nse_symbols.csv")
    
    return nse_stocks

if __name__ == "__main__":
    symbols = download_instrument_master()
    print(symbols.head(10))