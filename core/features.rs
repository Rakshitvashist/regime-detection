pub fn compute_returns(closes: &Vec<f64>, window: usize) -> Vec<f64> {
    let mut returns = vec![0.0; closes.len()];

    for i in window..closes.len() {
        returns[i] = (closes[i] / closes[i - window]) - 1.0;
    }

    returns
}

pub fn compute_volatility(returns: &Vec<f64>, window: usize) -> Vec<f64> {
    let mut vol = vec![0.0; returns.len()];

    for i in window..returns.len() {
        let slice = &returns[i - window..i];

        let mean = slice.iter().sum::<f64>() / window as f64;
        let var = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / window as f64;

        vol[i] = var.sqrt() * (252.0_f64).sqrt();
    }

    vol
}
