pub fn pct_change(closes: &Vec<f64>, lag: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; closes.len()];

    for i in lag..closes.len() {
        out[i] = (closes[i] / closes[i - lag]) - 1.0;
    }

    out
}
