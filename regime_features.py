"""
regime_features.py
==================
Single source of truth for the ML feature pipeline.

Previously XGB_FEATURES, add_structural_features(), assign_regime() and the
confirmation checklist were copy-pasted into xgb_regime.py, predict.py,
predict_oos.py and export_model_metrics.py — and had already drifted
(`.iloc` vs `.loc`, different confidence thresholds). Every consumer now
imports from here so the training-time and inference-time feature
computation are guaranteed identical.
"""

import numpy as np
import pandas as pd

# -----------------------------
# CONSTANTS
# -----------------------------
# Single decision threshold used everywhere a regime is assigned from
# probabilities. Below this the bar is declared "sideways" (insufficient
# conviction). Previously this was 0.65 in predict.py but 0.5 in
# xgb_regime.py / predict_oos.py, so the same model produced different
# labels depending on which script wrote the JSON.
CONFIDENCE_TH = 0.50

# Separate, stricter gate used only for the display checklist's
# "ml_confidence" tick (not for the regime decision itself).
CHECKLIST_CONF = 0.65

LABELS = ["bear", "sideways", "bull"]
REGIME_TO_INT = {"bear": 0, "sideways": 1, "bull": 2}
INT_TO_REGIME = {0: "bear", 1: "sideways", 2: "bull"}

# Feature vector fed to XGBoost. Order matters — the model and scaler are
# pickled against this exact ordering.
XGB_FEATURES = [
    "ret_10d",
    "vol_21d", "vol_ratio",
    "skew_21d", "kurt_21d",
    "rsi_14", "rsi_21",
    "macd_h", "bb_pct", "ema_gap", "adx",
    "drawdown_21d",
    "mom_divergence",
    "vol_spike",
    "dist_from_252h",
    "tii_21",
    "vol_of_vol_21",
]


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the structural features the model needs from the base feature
    table produced by the Rust core. This is the ONLY implementation — do not
    copy it into individual scripts.

    The caller is expected to pass a frame sorted by date. We reset the index
    locally so the rolling-window label lookups are unambiguous.
    """
    df = df.copy().reset_index(drop=True)

    # Base volatility (only if the Rust core didn't already provide it).
    if "vol_21d" not in df.columns:
        df["vol_21d"] = df["close"].pct_change().rolling(21).std() * np.sqrt(252)

    rolling_max_21 = df["close"].rolling(21, min_periods=1).max()
    rolling_max_252 = df["close"].rolling(252, min_periods=1).max()

    df["drawdown_21d"] = (df["close"] - rolling_max_21) / rolling_max_21
    df["dist_from_252h"] = (df["close"] - rolling_max_252) / rolling_max_252

    if "ret_21d" not in df.columns:
        df["ret_21d"] = df["close"].pct_change(21)

    df["mom_divergence"] = df["ret_10d"] - df["ret_21d"]
    df["vol_spike"] = df["vol_21d"] / df["vol_21d"].rolling(63, min_periods=1).mean()

    # TII (Trend Intensity): fraction of the last 21 closes that sit above the
    # 21-day SMA as of the window's final bar. Use .loc (label-based) so the
    # lookup is correct regardless of how the frame is indexed.
    sma21 = df["close"].rolling(21, min_periods=1).mean()
    df["tii_21"] = df["close"].rolling(21).apply(
        lambda x: (x.values > sma21.loc[x.index[-1]]).sum() / 21.0, raw=False
    )

    # Vol of vol: instability of the volatility series itself.
    df["vol_of_vol_21"] = df["vol_21d"].rolling(21, min_periods=1).std()

    fix_cols = [
        "drawdown_21d", "dist_from_252h", "mom_divergence",
        "vol_spike", "tii_21", "vol_of_vol_21", "vol_21d", "ret_21d",
    ]
    df[fix_cols] = df[fix_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def assign_regime(probs, threshold: float = CONFIDENCE_TH) -> str:
    """Map a [bear, sideways, bull] probability vector to a regime label.

    The strongest class wins unless its probability is below `threshold`, in
    which case we fall back to "sideways" (not enough conviction to call a
    directional regime).
    """
    idx = int(np.argmax(probs))
    if float(probs[idx]) < threshold:
        return "sideways"
    return LABELS[idx]


def calculate_checklist(row, probs):
    """Bull/bear confirmation checklist surfaced in the dashboard's
    Checklist tab. Pure technical cross-checks on top of the ML signal.
    """
    p_bear, _p_side, p_bull = probs

    bull_checks = {
        "momentum_21d": bool(row.get("ret_21d", 0) > 0.05),
        "trend_63d": bool(row.get("ret_63d", 0) > 0.08),
        "vol_stability": bool(row.get("vol_21d", 0) < row.get("vol_63d", 0)),
        "risk_cooling": bool(row.get("vol_ratio", 0) < 0.90),
        "tail_risk": bool(row.get("skew_21d", 0) > -0.5),
        "ema_cross": bool(row.get("ema_gap", 0) > 0),
        "macd_hist": bool(row.get("macd_h", 0) > 0),
        "rsi_healthy": bool(50 <= row.get("rsi_14", 0) <= 70),
        "ml_confidence": bool(p_bull >= CHECKLIST_CONF),
    }

    bear_checks = {
        "momentum_21d": bool(row.get("ret_21d", 0) < -0.05),
        "trend_63d": bool(row.get("ret_63d", 0) < -0.08),
        "panic_vol": bool(row.get("vol_21d", 0) > (row.get("vol_63d", 0) * 1.3)),
        "tail_crash": bool(row.get("skew_21d", 0) < -1.0),
        "ema_death_cross": bool(row.get("ema_gap", 0) < 0),
        "macd_hist_neg": bool(row.get("macd_h", 0) < 0),
        "rsi_weak": bool(row.get("rsi_14", 0) < 40),
        "ml_confidence": bool(p_bear >= CHECKLIST_CONF),
    }

    return {"bull": bull_checks, "bear": bear_checks}


def dashboard_metrics(row) -> dict:
    """Common set of support metrics every served regime row carries so the
    dashboard charts have consistent fields across in-sample and OOS files.
    """
    def g(key, default):
        val = row.get(key, default)
        return float(val) if pd.notna(val) else float(default)

    return {
        "ret_21d": g("ret_21d", 0.0),
        "ret_63d": g("ret_63d", 0.0),
        "vol_21d": g("vol_21d", 0.0),
        "vol_63d": g("vol_63d", 0.0),
        "vol_ratio": g("vol_ratio", 1.0),
        "skew_21d": g("skew_21d", 0.0),
        "rsi_14": g("rsi_14", 50.0),
        "macd_h": g("macd_h", 0.0),
        "ema_gap": g("ema_gap", 0.0),
        "drawdown_21d": g("drawdown_21d", 0.0),
        "tii_21": g("tii_21", 0.0),
        "vol_of_vol_21": g("vol_of_vol_21", 0.0),
        "dist_from_252h": g("dist_from_252h", 0.0),
    }
