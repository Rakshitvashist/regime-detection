import pandas as pd
import os

files = {
    "NIFTY": "NSE_NIFTY, 1D.csv",
    "NIFTY_500": "NSE_CNX500, 1D.csv",
    "CRUDE": "MCX_CRUDEOIL1!, 1D.csv",
    "WTI": "CFI_WTI, 1D.csv",
    "USDINR": "FX_IDC_USDINR, 1D.csv"
}

dfs = []
for name, path in files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time']).dt.date
        df = df[['time', 'close']].rename(columns={'close': name})
        dfs.append(df)

# Merge all on time
final_df = dfs[0]
for i in range(1, len(dfs)):
    final_df = pd.merge(final_df, dfs[i], on='time', how='inner')

# Filter since 2019
final_df = final_df[final_df['time'] >= pd.to_datetime('2019-01-01').date()]

# Calculate Correlation
corr = final_df.drop(columns='time').corr()

print("\n--- Correlation Matrix (Since 2019) ---")
print(corr)

# Save result for later
final_df.to_csv("research/asset_correlations.csv", index=False)
