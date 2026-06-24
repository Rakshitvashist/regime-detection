"""
fetch_intraday.py
=================
SEPARATE intraday data fetch for the Intraday track (Nifty50 only).

The daily pipeline (fetch_data.py) saves YYYY-MM-DD bars, which cannot support
intraday forecasting. This script pulls true intraday bars for ^NSEI from
yfinance and writes them to their own files so the intraday work stays isolated
from the daily weekly/monthly pipeline.

yfinance intraday history limits (as of writing):
  * 15m interval -> ~60 days of history
  * 60m interval -> ~730 days (~2 years) of history

Output (kept separate from the *, 1D.csv daily files):
  NSE_NIFTY_intraday_15m.csv
  NSE_NIFTY_intraday_60m.csv
Columns match the daily format (time, open, high, low, close, EMA, Volume) but
`time` carries the full intraday timestamp.
"""
import os
import sys

import pandas as pd
import yfinance as yf

TICKER = "^NSEI"          # Nifty 50
HERE = os.path.dirname(os.path.abspath(__file__))

# (interval, period, output filename)
JOBS = [
    ("15m", "60d", "NSE_NIFTY_intraday_15m.csv"),
    ("60m", "730d", "NSE_NIFTY_intraday_60m.csv"),
]


def fetch(interval, period, out_name):
    print(f"Fetching {TICKER} {interval} bars (period={period})...")
    df = yf.download(TICKER, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        print(f"  WARNING: no data returned for {interval}")
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    # the datetime column is 'Datetime' for intraday, 'Date' for daily
    tcol = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={tcol: "time", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close", "Volume": "Volume"})
    if "Volume" not in df.columns:
        df["Volume"] = 0
    df = df[["time", "open", "high", "low", "close", "Volume"]].copy()
    df.insert(5, "EMA", 0)               # match daily column layout
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    out_path = os.path.join(HERE, out_name)
    df.to_csv(out_path, index=False)
    print(f"  Saved {out_name}: {len(df)} bars | {df['time'].min()} -> {df['time'].max()}")
    return out_path


def main():
    ok = 0
    for interval, period, out_name in JOBS:
        try:
            if fetch(interval, period, out_name):
                ok += 1
        except Exception as e:
            print(f"  FAILED {interval}: {type(e).__name__}: {e}")
    print(f"\nDone: {ok}/{len(JOBS)} intraday files written.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
