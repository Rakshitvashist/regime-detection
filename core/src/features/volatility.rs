pub fn rolling_vol(returns: &Vec<f64>, window: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; returns.len()];
    
    if window < 2 { return out; }

    for i in window..returns.len() {
        let slice = &returns[i - window..i];

        let mean = slice.iter().sum::<f64>() / window as f64;

        let var = slice
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (window - 1) as f64; // Correct: Sample Variance (N-1)

        out[i] = var.sqrt() * (252.0_f64).sqrt();
    }

    out
}
