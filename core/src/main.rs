mod types;
mod data_engine;
mod feature_engine;
mod features;

use crate::data_engine::load_folder;
use crate::feature_engine::{
    build_features,
    save_to_json,
    save_latest_snapshot,
    save_regime_history
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        println!("Usage: cargo run -- <DATA_FOLDER_PATH> <SYMBOL>");
        println!("Example: cargo run -- \"D:/locale/data/NIFTY/spot\" NIFTY");
        return;
    }

    let folder_path = &args[1];
    let symbol_name = &args[2];
    let start_date_arg = args.get(3);

    println!("⚡ Initializing Regime Analysis for: {}", symbol_name);
    
    // 🔥 Direct CSV vs Folder check
    let data = if folder_path.ends_with(".csv") {
        crate::data_engine::load_daily_file(folder_path)
    } else {
        load_folder(folder_path)
    };

    if data.is_empty() {
        println!("❌ Error: No data loaded from {}", folder_path);
        return;
    }

    println!("Loaded {} bars for {}", data.len(), symbol_name);

    let mut features = build_features(&data, symbol_name);

    // Filter by date if provided
    if let Some(start_date_str) = start_date_arg {
        match chrono::NaiveDate::parse_from_str(start_date_str, "%Y-%m-%d") {
            Ok(start_date) => {
                let initial_count = features.len();
                features.retain(|f| f.date >= start_date);
                println!("✂️ Truncated features from {} to {} rows (Start Date: {})", initial_count, features.len(), start_date);
            },
            Err(e) => {
                println!("⚠️ Warning: Failed to parse start date '{}': {}. Using full history.", start_date_str, e);
            }
        }
    }

    // Save with symbol prefix
    save_to_json(&features, symbol_name);
    
    save_latest_snapshot(&features, symbol_name);
    save_regime_history(&features, symbol_name);

    println!("✅ Instrument Analysis Complete for {} 🚀", symbol_name);
}