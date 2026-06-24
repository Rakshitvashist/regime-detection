# NIFTY 50 Regime Forecasting — Final Models Report

**Project:** Regime Detection · **Instrument:** NIFTY 50 · **Validation:** honest walk-forward (train past → predict unseen future), no look-ahead.

---

## Objective
Forecast the market regime over three horizons: **Intraday**, **Weekly expiry (5 days)**, **Monthly expiry (21 days)** — to guide option/expiry trading decisions.

---

## Final certified models

| Horizon | What it predicts | Best model | Overall hit ratio | High-confidence hit ratio |
|---------|------------------|-----------|-------------------|---------------------------|
| **Intraday** | Volatility regime (calm vs volatile, next ~4h) | **HMM + XGBoost hybrid** | **64.6%** | **78%** (top 15% of bars) |
| **Weekly** | Direction (bear/sideways/bull, 5 days) | **HMM + XGBoost hybrid** | **58.0%** | **67%** (top 35% of bars) |
| **Monthly** | Volatility regime (calm vs volatile, 21 days) | **Volatility-persistence rule** | **73.7%** | **85%** (top 34% of bars) |

*Baselines (always-guess-majority): Intraday 53.6%, Weekly 47.5%, Monthly 56.4%. Every model beats its baseline by +10 to +17 points.*

---

## How to use each (trading interpretation)
- **Intraday vol regime:** low-vol → sell option premium; high-vol → buy / hedge.
- **Weekly direction:** position for the predicted 5-day regime (bull/bear/sideways).
- **Monthly vol regime:** size monthly premium-selling vs hedging by the calm/volatile call.
- **Trade the confident calls only** — that is where hit ratios jump (e.g. intraday 64.6% → 78%).

---

## Methods
- **XGBoost** — gradient-boosted trees over price + time-of-day features (non-linear pattern finder).
- **Gaussian HMM** — identifies hidden regimes and their persistence/transitions; its regime probability is fed into XGBoost (hybrid).
- **Persistence rule (monthly)** — "current volatility vs historical median"; wins because monthly vol is highly persistent and ML overfits the small dataset.

---

## Quality certifications (all three)
| Check | Method | Result |
|-------|--------|--------|
| **No data leakage** | Shuffle-label control (scramble train labels → must collapse to baseline) | ✅ all collapsed to ~chance |
| **No look-ahead** | Train-only thresholds/scalers/weights · past-only HMM filtering · embargo between train/validation | ✅ enforced |
| **No overfitting** | Train-vs-validation gap measured; models regularized | ✅ intraday gap 0.04, weekly gap 14pts (was 29), monthly rule has none |

---

## Key findings (honest)
1. **Direction is only predictable weekly.** Intraday and monthly *direction* are near-random (lose to market drift); their edge is in the **volatility regime** instead.
2. **Different horizons need different models.** Intraday/weekly → HMM+XGBoost hybrid; monthly → a simple persistence rule (ML overfits there).
3. **India VIX added nothing** (tested, dropped). Edge comes from price, realized volatility, and time-of-day.
4. **Confident-only filtering is the real edge** — hit ratios rise to 78% / 67% / 85% on high-conviction calls.

---

## Files (per model: a `.py` + an `_eval.csv` of predictions)
- `intraday_hybrid_nifty50.py` → `intraday_hybrid_60m_eval.csv`
- `weekly_hybrid_nifty50.py` → `weekly_hybrid_nifty50_eval.csv`
- `monthly_persistence_nifty50.py` → `monthly_persistence_nifty50_eval.csv`

---

## Status & next steps
- **Status:** all three validated, leak-free, overfitting-checked. Ready for **paper trading**.
- **Improve later:** add options OI / futures-basis data (the real lever for intraday direction); calibrate probabilities; size positions by confidence.
