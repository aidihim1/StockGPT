# merge_data.py -- merges returns_old.csv (Yahoo, pre-2020) + returns_dataset.csv (Angel One, 2021+)

import pandas as pd
import numpy as np

print("Loading returns_old.csv ...")
df_old = pd.read_csv("returns_old.csv", index_col=0, parse_dates=True)
print(f"  Old data shape: {df_old.shape} | {df_old.index[0].date()} to {df_old.index[-1].date()}")

print("Loading returns_dataset.csv ...")
df_new = pd.read_csv("returns_dataset.csv", index_col=0, parse_dates=True)
print(f"  New data shape: {df_new.shape} | {df_new.index[0].date()} to {df_new.index[-1].date()}")

# Winsorize extreme outliers (Yahoo data has bad splits/adjustments beyond +-100%)
before_old = (df_old.abs() > 1.0).sum().sum()
before_new = (df_new.abs() > 1.0).sum().sum()
df_old = df_old.clip(lower=-1.0, upper=1.0)
df_new = df_new.clip(lower=-1.0, upper=1.0)
print(f"\nWinsorized {before_old} values in old data and {before_new} in new data (clipped to +-100%)")

# Concat on rows -- outer join keeps all columns from both files
# Old data will have NaN for the 934 Angel-only stocks (correct -- they didn't exist pre-2021)
print("\nMerging datasets ...")
combined = pd.concat([df_old, df_new], axis=0, join="outer")
combined.sort_index(inplace=True)

print(f"\nCombined shape : {combined.shape}")
print(f"Date range     : {combined.index[0].date()} to {combined.index[-1].date()}")
print(f"Total stocks   : {combined.shape[1]}")
print(f"NaN fraction   : {combined.isna().mean().mean():.1%}")

# Save
combined.to_csv("combined_returns.csv")
print(f"\nSaved combined_returns.csv")
print("Done.")
