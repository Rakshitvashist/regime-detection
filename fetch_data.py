import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# Map the TradingView-style filenames to Yahoo Finance tickers
TICKER_MAP = {
    "NSE_NIFTY": "^NSEI",
    "NSE_BANKNIFTY": "^NSEBANK",
    "NSE_CNX500": "^CRSLDX",
    "FX_IDC_USDINR": "INR=X",
    "CFI_WTI": "CL=F",        # WTI crude (NYMEX)
    # MCX crude isn't reliably on yfinance. It tracks WTI, but mapping it to WTI
    # too made "NIFTY -> Crude" and "NIFTY -> WTI" identical OOS tests. Use Brent
    # (a distinct crude benchmark) so the two generalization tests differ.
    "MCX_CRUDEOIL1!": "BZ=F",  # Brent crude (ICE) as a distinct crude proxy
}

START_DATE = "2010-01-01"

def fetch_and_save(filename_prefix, ticker):
    print(f"Fetching {ticker} for {filename_prefix}...")
    
    # Download data
    df = yf.download(ticker, start=START_DATE, progress=False)
    
    if df.empty:
        print(f"Warning: No data found for {ticker}")
        return

    # yfinance sometimes returns MultiIndex columns. Flatten if necessary.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # We need the format: time,open,high,low,close,EMA,Volume
    # data_engine.rs relies strictly on the first 5 columns: Date, Open, High, Low, Close.
    
    # Reset index to get 'Date' as a column
    df = df.reset_index()
    
    # Rename 'Date' to 'time' to match the original TradingView CSV format
    df.rename(columns={'Date': 'time', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'Volume'}, inplace=True)
    
    # Ensure datetime format is strictly YYYY-MM-DD
    df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
    
    # Select columns in the expected order
    cols_to_keep = ['time', 'open', 'high', 'low', 'close', 'Volume']
    # If Volume is missing (e.g., for some forex pairs), add it as 0
    if 'Volume' not in df.columns:
        df['Volume'] = 0
        
    df = df[cols_to_keep]
    
    # Add a dummy 'EMA' column to exactly match the TradingView structure if needed by other scripts
    # (data_engine.rs doesn't strictly need it, but we insert it at the correct position)
    df.insert(5, 'EMA', 0)
    
    # Save to CSV
    filename = f"{filename_prefix}, 1D.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {filename} with {len(df)} rows.")

def main():
    for filename_prefix, ticker in TICKER_MAP.items():
        try:
            fetch_and_save(filename_prefix, ticker)
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")

if __name__ == "__main__":
    main()
