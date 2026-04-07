use crate::types::DailyBar;
use chrono::NaiveDate;

// -----------------------------
// FEATURE STRUCT
// -----------------------------
#[derive(Debug)]
pub struct FeatureRow {
    pub date: NaiveDate,
    pub close: f64,
    pub ret_21d: Option<f64>,
    pub ret_63d: Option<f64>,
    pub vol_21d: Option<f64>,
    pub vol_63d: Option<f64>,
    pub vol_ratio: Option<f64>,
    pub skew_21d: Option<f64>,
    pub ret_z21: Option<f64>,
}

// -----------------------------
// MAIN FEATURE FUNCTION
// -----------------------------
pub fn compute_features(data: &Vec<DailyBar>) -> Vec<FeatureRow> {
    let closes: Vec<f64> = data.iter().map(|x| x.close).collect();

    let ret_21d = compute_returns(&closes, 21);
    let ret_63d = compute_returns(&closes, 63);

    let vol_21d = compute_volatility(&closes, 21);
    let vol_63d = compute_volatility(&closes, 63);

    let vol_ratio = compute_ratio(&vol_21d, &vol_63d);

    let skew_21d = rolling_skew(&closes, 21);
    let ret_z21 = rolling_zscore(&ret_21d, 63);

    let mut output = Vec::with_capacity(data.len());

    for i in 0..data.len() {
        output.push(FeatureRow {
            date: data[i].date,
            close: data[i].close,
            ret_21d: ret_21d[i],
            ret_63d: ret_63d[i],
            vol_21d: vol_21d[i],
            vol_63d: vol_63d[i],
            vol_ratio: vol_ratio[i],
            skew_21d: skew_21d[i],
            ret_z21: ret_z21[i],
        });
    }

    output
}

// -----------------------------
// RETURNS
// -----------------------------
fn compute_returns(prices: &Vec<f64>, window: usize) -> Vec<Option<f64>> {
    let mut result = vec![None; prices.len()];

    for i in window..prices.len() {
        result[i] = Some((prices[i] / prices[i - window]) - 1.0);
    }

    result
}

// -----------------------------
// VOLATILITY
// -----------------------------
fn compute_volatility(prices: &Vec<f64>, window: usize) -> Vec<Option<f64>> {
    let mut returns = vec![0.0; prices.len()];

    for i in 1..prices.len() {
        returns[i] = (prices[i] / prices[i - 1]) - 1.0;
    }

    let mut result = vec![None; prices.len()];

    for i in window..prices.len() {
        let slice = &returns[i - window..i];

        let mean = slice.iter().sum::<f64>() / window as f64;

        let var = slice.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / window as f64;

        result[i] = Some(var.sqrt() * (252.0_f64).sqrt());
    }

    result
}

// -----------------------------
// RATIO
// -----------------------------
fn compute_ratio(a: &Vec<Option<f64>>, b: &Vec<Option<f64>>) -> Vec<Option<f64>> {
    let mut result = vec![None; a.len()];

    for i in 0..a.len() {
        if let (Some(x), Some(y)) = (a[i], b[i]) {
            if y != 0.0 {
                result[i] = Some(x / y);
            }
        }
    }

    result
}

// -----------------------------
// SKEWNESS
// -----------------------------
fn rolling_skew(data: &Vec<f64>, window: usize) -> Vec<Option<f64>> {
    let mut result = vec![None; data.len()];

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
            result[i] = Some(m3 / m2.powf(1.5));
        }
    }

    result
}

// -----------------------------
// Z-SCORE
// -----------------------------
fn rolling_zscore(data: &Vec<Option<f64>>, window: usize) -> Vec<Option<f64>> {
    let mut result = vec![None; data.len()];

    for i in window..data.len() {
        let slice: Vec<f64> = data[i - window..i]
            .iter()
            .filter_map(|&x| x)
            .collect();

        if slice.len() < window {
            continue;
        }

        let mean = slice.iter().sum::<f64>() / window as f64;

        let var = slice.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / window as f64;

        let std = var.sqrt();

        if std > 0.0 {
            if let Some(val) = data[i] {
                result[i] = Some((val - mean) / std);
            }
        }
    }

    result
}