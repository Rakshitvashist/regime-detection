use crate::types::DailyBar;

pub fn extract_closes(data: &Vec<DailyBar>) -> Vec<f64> {
    data.iter().map(|x| x.close).collect()
}
