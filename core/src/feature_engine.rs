use crate::types::{DailyBar, FeatureRow};
use crate::features::returns::pct_change;
use crate::features::volatility::rolling_vol;
use crate::features::utils::extract_closes;

use std::fs::File;
use std::io::Write;

// -----------------------------
// SAVE FULL FEATURES JSON
// -----------------------------
pub fn save_to_json(features: &Vec<FeatureRow>, symbol: &str) {
    let json = serde_json::to_string_pretty(features).unwrap();
    let filename = format!("../output/features_{}.json", symbol);
    let mut file = File::create(filename).unwrap();
    file.write_all(json.as_bytes()).unwrap();
}

// -----------------------------
// BUILD FEATURES (FULL 19)
// -----------------------------
pub fn build_features(data: &Vec<DailyBar>, symbol: &str) -> Vec<FeatureRow> {
    let closes = extract_closes(data);

    // -----------------------------
    // MOMENTUM
    // -----------------------------
    let ret_1d = pct_change(&closes, 1);
    let ret_5d = pct_change(&closes, 5);
    let ret_10d = pct_change(&closes, 10);
    let ret_21d = pct_change(&closes, 21);
    let ret_42d = pct_change(&closes, 42);
    let ret_63d = pct_change(&closes, 63);
    let ret_126d = pct_change(&closes, 126);

    // -----------------------------
    // VOLATILITY
    // -----------------------------
    let vol_10d = rolling_vol(&ret_1d, 10);
    let vol_21d = rolling_vol(&ret_1d, 21);
    let vol_63d = rolling_vol(&ret_1d, 63);



    // -----------------------------
    // DISTRIBUTION
    // -----------------------------
    let skew_21d = rolling_skew(&closes, 21);
    let kurt_21d = rolling_kurt(&closes, 21);

    // -----------------------------
    // TECHNICAL
    // -----------------------------
    let rsi_14 = compute_rsi(&closes, 14);
    let rsi_21 = compute_rsi(&closes, 21);
    let macd_h = compute_macd_hist(&closes);
    let bb_pct = compute_bb_pct(&closes, 20);
    let ema_gap = compute_ema_gap(&closes, 20, 60);
    let adx = compute_adx(data, 14);

    // -----------------------------
    // Z-SCORE
    // -----------------------------
    let ret_z21 = rolling_zscore(&ret_21d, 126);

    // -----------------------------
    // NICHE
    // -----------------------------
    let max_252 = rolling_max(&closes, 252);
    let sma_21 = rolling_sma(&closes, 21);
    let vov_21 = rolling_std(&vol_21d, 21);

    let mut features = Vec::with_capacity(data.len());

    for i in 0..data.len() {
        let vr = if vol_63d[i].is_nan() || vol_63d[i] == 0.0 {
            f64::NAN
        } else {
            vol_21d[i] / vol_63d[i]
        };

        features.push(FeatureRow {
            date: data[i].date,
            symbol: symbol.to_string(),
            close: closes[i],

            // momentum
            ret_1d: ret_1d[i],
            ret_5d: ret_5d[i],
            ret_10d: ret_10d[i],
            ret_21d: ret_21d[i],
            ret_42d: ret_42d[i],
            ret_63d: ret_63d[i],
            ret_126d: ret_126d[i],

            // volatility
            vol_10d: vol_10d[i],
            vol_21d: vol_21d[i],
            vol_63d: vol_63d[i],
            vol_ratio: vr,

            // distribution
            skew_21d: skew_21d[i],
            kurt_21d: kurt_21d[i],

            // technical
            rsi_14: rsi_14[i],
            rsi_21: rsi_21[i],
            macd_h: macd_h[i],
            bb_pct: bb_pct[i],
            ema_gap: ema_gap[i],
            adx: adx[i],

            // mean reversion
            ret_z21: ret_z21[i],

            // niche
            dist_from_252h: if max_252[i] > 0.0 { (closes[i] - max_252[i]) / max_252[i] } else { 0.0 },
            tii_21: {
                if i < 21 { 0.0 } else {
                    let mut count = 0;
                    for j in (i-20)..=i {
                       if closes[j] > sma_21[i] { count += 1; }
                    }
                    count as f64 / 21.0
                }
            },
            vol_of_vol_21: vov_21[i],
        });
    }

    features
}

// -----------------------------
// 🔥 RATIO
// -----------------------------
fn compute_ratio(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let mut out = vec![f64::NAN; a.len()];
    for i in 0..a.len() {
        if b[i] != 0.0 {
            out[i] = a[i] / b[i];
        }
    }
    out
}

// -----------------------------
// SKEWNESS
// -----------------------------
fn rolling_skew(data: &Vec<f64>, window: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; data.len()];

    for i in window..data.len() {
        let slice = &data[i - window..i];
        let mean = slice.iter().sum::<f64>() / window as f64;

        let mut m2 = 0.0;
        let mut m3 = 0.0;

        for &x in slice {
            let d = x - mean;
            m2 += d.powi(2);
            m3 += d.powi(3);
        }

        m2 /= window as f64;
        m3 /= window as f64;

        if m2 > 0.0 {
            result[i] = m3 / m2.powf(1.5);
        }
    }

    result
}

// -----------------------------
// KURTOSIS
// -----------------------------
fn rolling_kurt(data: &Vec<f64>, window: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];

    for i in window..data.len() {
        let slice = &data[i - window..i];
        let mean = slice.iter().sum::<f64>() / window as f64;

        let mut m2 = 0.0;
        let mut m4 = 0.0;

        for &x in slice {
            let d = x - mean;
            m2 += d.powi(2);
            m4 += d.powi(4);
        }

        m2 /= window as f64;
        m4 /= window as f64;

        if m2 > 0.0 {
            out[i] = m4 / m2.powi(2);
        }
    }

    out
}

// -----------------------------
// RSI
// -----------------------------
fn compute_rsi(prices: &Vec<f64>, period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; prices.len()];

    for i in period..prices.len() {
        let mut gain = 0.0;
        let mut loss = 0.0;

        for j in (i - period + 1)..=i {
            let diff = prices[j] - prices[j - 1];
            if diff > 0.0 {
                gain += diff;
            } else {
                loss -= diff;
            }
        }

        let rs = if loss == 0.0 { 100.0 } else { gain / loss };
        out[i] = 100.0 - (100.0 / (1.0 + rs));
    }

    out
}

// -----------------------------
// EMA
// -----------------------------
fn ema(data: &Vec<f64>, period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    let alpha = 2.0 / (period as f64 + 1.0);

    out[0] = data[0];

    for i in 1..data.len() {
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1];
    }

    out
}

// -----------------------------
// MACD HIST
// -----------------------------
fn compute_macd_hist(prices: &Vec<f64>) -> Vec<f64> {
    let ema12 = ema(prices, 12);
    let ema26 = ema(prices, 26);

    let mut macd = vec![f64::NAN; prices.len()];
    for i in 0..prices.len() {
        macd[i] = ema12[i] - ema26[i];
    }

    let signal = ema(&macd, 9);

    let mut hist = vec![f64::NAN; prices.len()];
    for i in 0..prices.len() {
        hist[i] = macd[i] - signal[i];
    }

    hist
}

// -----------------------------
// BB %
// -----------------------------
fn compute_bb_pct(prices: &Vec<f64>, window: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; prices.len()];

    for i in window..prices.len() {
        let slice = &prices[i - window..i];
        let mean = slice.iter().sum::<f64>() / window as f64;

        let std = (slice.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / window as f64).sqrt();

        let upper = mean + 2.0 * std;
        let lower = mean - 2.0 * std;

        if upper != lower {
            out[i] = (prices[i] - lower) / (upper - lower);
        }
    }

    out
}

// -----------------------------
// EMA GAP
// -----------------------------
fn compute_ema_gap(prices: &Vec<f64>, short: usize, long: usize) -> Vec<f64> {
    let ema_s = ema(prices, short);
    let ema_l = ema(prices, long);

    let mut out = vec![f64::NAN; prices.len()];

    for i in 0..prices.len() {
        if ema_l[i] != 0.0 {
            out[i] = (ema_s[i] - ema_l[i]) / ema_l[i];
        }
    }

    out
}

// -----------------------------
// ADX
// -----------------------------
fn compute_adx(data: &Vec<DailyBar>, window: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];

    for i in window..data.len() {
        let mut tr_sum = 0.0;

        for j in (i - window + 1)..=i {
            let high = data[j].high;
            let low = data[j].low;
            let prev_close = data[j - 1].close;

            let tr = (high - low)
                .max((high - prev_close).abs())
                .max((low - prev_close).abs());

            tr_sum += tr;
        }

        out[i] = tr_sum / window as f64;
    }

    out
}

// -----------------------------
// Z-SCORE
// -----------------------------
fn rolling_zscore(data: &Vec<f64>, window: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; data.len()];
    for i in window..data.len() {
        let slice = &data[i - window..i];
        let n = slice.iter().filter(|&&x| !x.is_nan()).count() as f64;
        if n < 2.0 { continue; }
        let mean = slice.iter().filter(|&&x| !x.is_nan()).sum::<f64>() / n;
        let var = slice.iter().filter(|&&x| !x.is_nan()).map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();
        if std > 0.0 {
            result[i] = (data[i] - mean) / std;
        }
    }
    result
}

fn rolling_max(data: &Vec<f64>, window: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    for i in 0..data.len() {
        let start = if i >= window { i - window + 1 } else { 0 };
        let slice = &data[start..=i];
        let mut m = slice[0];
        for &val in slice {
            if val > m { m = val; }
        }
        out[i] = m;
    }
    out
}

fn rolling_sma(data: &Vec<f64>, window: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    for i in 0..data.len() {
        let start = if i >= window { i - window + 1 } else { 0 };
        let slice = &data[start..=i];
        out[i] = slice.iter().sum::<f64>() / slice.len() as f64;
    }
    out
}

fn rolling_std(data: &Vec<f64>, window: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    for i in 0..data.len() {
        let start = if i >= window { i - window + 1 } else { 0 };
        let slice = &data[start..=i];
        let n = slice.len() as f64;
        if n < 2.0 { continue; }
        let mean = slice.iter().sum::<f64>() / n;
        let var = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        out[i] = var.sqrt();
    }
    out
}

// -----------------------------
// REGIME (UNCHANGED)
// -----------------------------
pub fn classify_regime(row: &FeatureRow) -> String {
    let trend = if row.ret_21d > 0.05 {
        "bull"
    } else if row.ret_21d < -0.05 {
        "bear"
    } else {
        "sideways"
    };

    let vol = if row.vol_ratio < 0.9 {
        "low_vol"
    } else if row.vol_ratio > 1.2 {
        "high_vol"
    } else {
        "normal_vol"
    };

    format!("{}_{}", trend, vol)
}

// -----------------------------
// SNAPSHOT
// -----------------------------
pub fn save_latest_snapshot(features: &Vec<FeatureRow>, symbol: &str) {
    let latest = features.last().unwrap();
    let regime = classify_regime(latest);

    let output = serde_json::json!({
        "date": latest.date,
        "symbol": latest.symbol,
        "close": latest.close,
        "ret_21d": latest.ret_21d,
        "vol_ratio": latest.vol_ratio,
        "regime": regime
    });

    let filename = format!("../output/regime_snapshot_{}.json", symbol);
    std::fs::write(
        filename,
        serde_json::to_string_pretty(&output).unwrap()
    ).unwrap();
}

// -----------------------------
// HISTORY
// -----------------------------
pub fn save_regime_history(features: &Vec<FeatureRow>, symbol: &str) {
    let mut out = Vec::new();

    for row in features {
        let regime = classify_regime(row);

        out.push(serde_json::json!({
            "date": row.date,
            "symbol": row.symbol,
            "close": row.close,
            "ret_21d": row.ret_21d,
            "vol_ratio": row.vol_ratio,
            "regime": regime
        }));
    }

    let filename = format!("../output/regime_history_{}.json", symbol);
    std::fs::write(
        filename,
        serde_json::to_string_pretty(&out).unwrap()
    ).unwrap();
}