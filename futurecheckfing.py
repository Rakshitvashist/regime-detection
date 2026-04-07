import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, log_loss

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "output/regime_ml_21d_v2.json"   # your file
FUTURES_PATH = "NSE_NIFTY1!, 1D.csv"        # your intraday/daily futures OHLC

FORWARD_DAYS = 21
BULL_TH = 0.02
BEAR_TH = -0.02


# ==============================
# LOAD DATA
# ==============================
df_model = pd.read_json(MODEL_PATH)
df_model['date'] = pd.to_datetime(df_model['date'])

df_fut = pd.read_csv(FUTURES_PATH)
df_fut['date'] = pd.to_datetime(df_fut['time'])

# If intraday → convert to daily close
if 'time' in df_fut.columns:
    df_fut = df_fut.sort_values(['date', 'time'])
    df_fut = df_fut.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()

# rename for clarity
df_fut.rename(columns={'close': 'close_fut'}, inplace=True)

# ==============================
# MERGE
# ==============================
df = df_model.merge(df_fut[['date', 'close_fut']], on='date', how='inner')
df = df.sort_values('date').reset_index(drop=True)

# ==============================
# REMOVE WARMUP
# ==============================
df = df[df['regime'].notna()].copy()
df = df[df['confidence'] > 0].copy()

print(f"Valid rows: {len(df)}")

# ==============================
# CREATE ACTUAL REGIME (FUTURES BASED)
# ==============================
df['fwd_ret_21d'] = df['close_fut'].shift(-FORWARD_DAYS) / df['close_fut'] - 1

def classify(x):
    if x > BULL_TH:
        return 'bull'
    elif x < BEAR_TH:
        return 'bear'
    else:
        return 'sideways'

df['regime_actual'] = df['fwd_ret_21d'].apply(classify)

# drop last rows where forward return unavailable
df = df.dropna(subset=['fwd_ret_21d'])

# ==============================
# ACCURACY
# ==============================
acc = (df['regime'] == df['regime_actual']).mean()
print("\n=== Accuracy ===")
print(f"{acc:.4f}")

# ==============================
# CONFUSION MATRIX
# ==============================
print("\n=== Confusion Matrix ===")
print(confusion_matrix(df['regime_actual'], df['regime']))

print("\n=== Classification Report ===")
print(classification_report(df['regime_actual'], df['regime']))

# ==============================
# LOG LOSS (probabilistic quality)
# ==============================
# map labels
label_map = {'bear': 0, 'sideways': 1, 'bull': 2}
df['y_true'] = df['regime_actual'].map(label_map)

probs = df[['prob_bear', 'prob_sideways', 'prob_bull']].values

try:
    ll = log_loss(df['y_true'], probs)
    print("\n=== Log Loss ===")
    print(ll)
except:
    print("\nLog loss failed (check probabilities)")

# ==============================
# PnL STRATEGY (DISCRETE)
# ==============================
def position(row):
    if row['regime'] == 'bull':
        return 1
    elif row['regime'] == 'bear':
        return -1
    else:
        return 0

df['pos'] = df.apply(position, axis=1)

df['daily_ret'] = df['close_fut'].pct_change()
df['strategy_ret'] = df['pos'].shift(1) * df['daily_ret']

df = df.dropna()

# ==============================
# METRICS
# ==============================
sharpe = df['strategy_ret'].mean() / df['strategy_ret'].std() * np.sqrt(252)
cagr = (1 + df['strategy_ret']).prod() ** (252/len(df)) - 1
cum = (1 + df['strategy_ret']).cumprod()
max_dd = (cum.cummax() - cum).max()

print("\n=== Strategy Metrics ===")
print(f"Sharpe: {sharpe:.2f}")
print(f"CAGR: {cagr:.2%}")
print(f"Max DD: {max_dd:.2%}")

# ==============================
# CONTINUOUS SIGNAL (ADVANCED)
# ==============================
df['expected_signal'] = (
    df['prob_bull'] * 1 +
    df['prob_bear'] * -1 +
    df['prob_sideways'] * 0
)

df['strategy_ret_prob'] = df['expected_signal'].shift(1) * df['daily_ret']

sharpe_prob = df['strategy_ret_prob'].mean() / df['strategy_ret_prob'].std() * np.sqrt(252)

print("\n=== Probabilistic Strategy Sharpe ===")
print(f"{sharpe_prob:.2f}")

# ==============================
# CONFIDENCE BUCKET ANALYSIS
# ==============================
df['conf_bucket'] = pd.qcut(df['confidence'], 5)

print("\n=== Confidence vs Return ===")
print(df.groupby('conf_bucket')['fwd_ret_21d'].mean())

# ==============================
# SAVE OUTPUT
# ==============================
df.to_csv("evaluation_output.csv", index=False)

print("\nSaved: evaluation_output.csv")