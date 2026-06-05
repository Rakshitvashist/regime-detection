use chrono::NaiveDate;
use serde::Serialize;

// -----------------------------
// DAILY BAR
// -----------------------------
#[derive(Debug)]
pub struct DailyBar {
    pub date: NaiveDate,
    #[allow(dead_code)]
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

// -----------------------------
// FEATURE ROW (FULL 19 FEATURES)
// -----------------------------
#[derive(Debug, Serialize)]
pub struct FeatureRow {
    pub date: NaiveDate,
    pub symbol: String,
    pub close: f64,

    // -----------------------------
    // MOMENTUM
    // -----------------------------
    pub ret_1d: f64,
    pub ret_5d: f64,
    pub ret_10d: f64,
    pub ret_21d: f64,
    pub ret_42d: f64,
    pub ret_63d: f64,
    pub ret_126d: f64,

    // -----------------------------
    // VOLATILITY
    // -----------------------------
    pub vol_10d: f64,
    pub vol_21d: f64,
    pub vol_63d: f64,
    pub vol_ratio: f64,

    // -----------------------------
    // DISTRIBUTION
    // -----------------------------
    pub skew_21d: f64,
    pub kurt_21d: f64,

    // -----------------------------
    // TECHNICAL
    // -----------------------------
    pub rsi_14: f64,
    pub rsi_21: f64,
    pub macd_h: f64,
    pub bb_pct: f64,
    pub ema_gap: f64,
    pub adx: f64,

    // -----------------------------
    // MEAN REVERSION
    // -----------------------------
    pub ret_z21: f64,

    // NOTE: the "niche" structural features (dist_from_252h, tii_21,
    // vol_of_vol_21) used to be emitted here too, but the Python pipeline
    // (regime_features.add_structural_features) recomputes them from `close`
    // and silently overwrote these values. They are now computed in exactly
    // one place (Python) to remove the duplicate, drift-prone definitions.
}