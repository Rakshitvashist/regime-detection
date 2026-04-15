use crate::types::DailyBar;
use chrono::NaiveDate;
use std::fs;


pub fn load_daily_file(path: &str) -> Vec<DailyBar> {
    let mut data = Vec::new();
    let file_data = std::fs::read_to_string(path).expect("Failed to read daily CSV file");

    for (i, line) in file_data.lines().enumerate() {
        if i == 0 || line.trim().is_empty() { continue; }

        let parts: Vec<&str> = line.split(',').collect();
        // Expected columns: time, open, high, low, close
        if parts.len() < 5 { continue; }

        let date = match NaiveDate::parse_from_str(parts[0], "%Y-%m-%d") {
            Ok(d) => d,
            Err(_) => continue,
        };

        let o = parts[1].parse::<f64>().ok();
        let h = parts[2].parse::<f64>().ok();
        let l = parts[3].parse::<f64>().ok();
        let c = parts[4].parse::<f64>().ok();

        if let (Some(o), Some(h), Some(l), Some(c)) = (o, h, l, c) {
            data.push(DailyBar {
                date,
                open: o,
                high: h,
                low: l,
                close: c,
            });
        }
    }

    data.sort_by_key(|x| x.date);
    data
}

pub fn load_folder(path: &str) -> Vec<DailyBar> {
    let mut data = Vec::new();

    let paths = fs::read_dir(path).expect("Failed to read directory");

    for entry in paths {
        let path = entry.unwrap().path();

        if path.extension().and_then(|s| s.to_str()) != Some("csv") {
            continue;
        }

        // 🔥 Extract date from filename
        let filename = path.file_stem().unwrap().to_str().unwrap();
        let date = match NaiveDate::parse_from_str(filename, "%Y-%m-%d") {
            Ok(d) => d,
            Err(_) => continue,
        };

        let file_data = match std::fs::read_to_string(&path) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let mut open = None;
        let mut high = f64::MIN;
        let mut low = f64::MAX;
        let mut close = None;

        for (i, line) in file_data.lines().enumerate() {
            if i == 0 { continue; }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 7 { continue; }

            let o = parts[3].parse::<f64>().ok();
            let h = parts[4].parse::<f64>().ok();
            let l = parts[5].parse::<f64>().ok();
            let c = parts[6].parse::<f64>().ok();

            if o.is_none() || h.is_none() || l.is_none() || c.is_none() {
                continue;
            }

            let o = o.unwrap();
            let h = h.unwrap();
            let l = l.unwrap();
            let c = c.unwrap();

            // first valid open
            if open.is_none() {
                open = Some(o);
            }

            high = high.max(h);
            low = low.min(l);
            close = Some(c);
        }

        if let (Some(o), Some(c)) = (open, close) {
            data.push(DailyBar {
                date,
                open: o,
                high,
                low,
                close: c,
            });
        }
    }

    data.sort_by_key(|x| x.date);

    data
}