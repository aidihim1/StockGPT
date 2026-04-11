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
    
    # Filter: only NSE exchange, only equity stocks (not futures/options/ETFs)
    # 'expiry' is empty for equity stocks (derivatives have an expiry date)
    nse_stocks = df[
        (df["exch_seg"] == "NSE") &          # NSE exchange only
        (df["instrumenttype"] == "")  &       # equity (not F&O)
        (df["symbol"].str.endswith("-EQ"))     # equity suffix
    ].copy()
    
    # Keep only the columns we need
    nse_stocks = nse_stocks[["token", "symbol", "name", "exch_seg"]].reset_index(drop=True)
    
    # Clean symbol name (remove -EQ suffix for readability)
    nse_stocks["clean_symbol"] = nse_stocks["symbol"].str.replace("-EQ", "")
    
    nse_stocks.to_csv("nse_symbols.csv", index=False)
    print(f"Saved {len(nse_stocks)} NSE equity stocks to nse_symbols.csv")
    
    return nse_stocks

if __name__ == "__main__":
    symbols = download_instrument_master()
    print(symbols.head(10))