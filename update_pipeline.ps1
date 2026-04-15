# update_pipeline.ps1
# ---------------------------------------------------------
# Manual script to update data, retrain models, and deploy.
# ---------------------------------------------------------

$DataPath = "D:\locale\data"
$Python = ".\.venv\Scripts\python.exe"

Write-Host "--- Starting Regime Detection Update Pipeline ---" -ForegroundColor Cyan

# 1. NIFTY
Write-Host "`n[1/3] Processing NIFTY..." -ForegroundColor Yellow
cargo run --manifest-path core/Cargo.toml --release -- "NSE_NIFTY, 1D.csv" NIFTY "2019-01-01"
& $Python research/kmeans_regime.py --input output/features_NIFTY.json --output output/regime_clustered_NIFTY.json --symbol NIFTY
& $Python research/xgb_regime.py --input output/features_NIFTY.json --model-out output/xgb_model_NIFTY.pkl --symbol NIFTY
& $Python predict.py --input output/features_NIFTY.json --model output/xgb_model_NIFTY.pkl --symbol NIFTY

# 2. BANKNIFTY
Write-Host "`n[2/3] Processing BANKNIFTY..." -ForegroundColor Yellow
cargo run --manifest-path core/Cargo.toml --release -- "NSE_BANKNIFTY, 1D.csv" BANKNIFTY "2019-01-01"
& $Python research/kmeans_regime.py --input output/features_BANKNIFTY.json --output output/regime_clustered_BANKNIFTY.json --symbol BANKNIFTY
& $Python research/xgb_regime.py --input output/features_BANKNIFTY.json --model-out output/xgb_model_BANKNIFTY.pkl --symbol BANKNIFTY
& $Python predict.py --input output/features_BANKNIFTY.json --model output/xgb_model_BANKNIFTY.pkl --symbol BANKNIFTY

# 3. NIFTY_500
Write-Host "`n[3/3] Processing NIFTY_500..." -ForegroundColor Yellow
cargo run --manifest-path core/Cargo.toml --release -- "NSE_CNX500, 1D.csv" NIFTY_500 "2019-01-01"
& $Python research/kmeans_regime.py --input output/features_NIFTY_500.json --output output/regime_clustered_NIFTY_500.json --symbol NIFTY_500
& $Python research/xgb_regime.py --input output/features_NIFTY_500.json --model-out output/xgb_model_NIFTY_500.pkl --symbol NIFTY_500
& $Python predict.py --input output/features_NIFTY_500.json --model output/xgb_model_NIFTY_500.pkl --symbol NIFTY_500

# 4. Cross-Asset OOS Experiments (Oil & Forex)
Write-Host "`n[4/4] Processing Cross-Asset OOS (Oil & Forex)..." -ForegroundColor Yellow

# Generate Features
cargo run --manifest-path core/Cargo.toml --release -- "MCX_CRUDEOIL1!, 1D.csv" CRUDE "2019-01-01"
cargo run --manifest-path core/Cargo.toml --release -- "CFI_WTI, 1D.csv" WTI "2019-01-01"
cargo run --manifest-path core/Cargo.toml --release -- "FX_IDC_USDINR, 1D.csv" USDINR "2019-01-01"

# Predictions: NIFTY model on Oil & FX
& $Python research/predict_oos.py --model output/xgb_model_NIFTY.pkl --input output/features_CRUDE.json --symbol CRUDE --trained-on NIFTY --json-out frontend/frontend/public/data/regime_NIFTY_on_CRUDE.json
& $Python research/predict_oos.py --model output/xgb_model_NIFTY.pkl --input output/features_WTI.json --symbol WTI --trained-on NIFTY --json-out frontend/frontend/public/data/regime_NIFTY_on_WTI.json
& $Python research/predict_oos.py --model output/xgb_model_NIFTY.pkl --input output/features_USDINR.json --symbol USDINR --trained-on NIFTY --json-out frontend/frontend/public/data/regime_NIFTY_on_USDINR.json

# Predictions: NIFTY_500 model on Oil & FX
& $Python research/predict_oos.py --model output/xgb_model_NIFTY_500.pkl --input output/features_CRUDE.json --symbol CRUDE --trained-on NIFTY_500 --json-out frontend/frontend/public/data/regime_NIFTY500_on_CRUDE.json
& $Python research/predict_oos.py --model output/xgb_model_NIFTY_500.pkl --input output/features_WTI.json --symbol WTI --trained-on NIFTY_500 --json-out frontend/frontend/public/data/regime_NIFTY500_on_WTI.json
& $Python research/predict_oos.py --model output/xgb_model_NIFTY_500.pkl --input output/features_USDINR.json --symbol USDINR --trained-on NIFTY_500 --json-out frontend/frontend/public/data/regime_NIFTY500_on_USDINR.json

# --- Deploy & Commit ---
& $Python research/predict_oos.py --model output/xgb_model_BANKNIFTY.pkl --input output/features_NIFTY.json --symbol NIFTY --trained-on BANKNIFTY --json-out frontend/frontend/public/data/regime_BANKNIFTY_on_NIFTY.json
& $Python research/predict_oos.py --model output/xgb_model_BANKNIFTY.pkl --input output/features_NIFTY_500.json --symbol NIFTY_500 --trained-on BANKNIFTY --json-out frontend/frontend/public/data/regime_BANKNIFTY_on_NIFTY500.json
& $Python research/predict_oos.py --model output/xgb_model_NIFTY_500.pkl --input output/features_NIFTY.json --symbol NIFTY --trained-on NIFTY_500 --json-out frontend/frontend/public/data/regime_NIFTY500_on_NIFTY.json
& $Python research/predict_oos.py --model output/xgb_model_NIFTY_500.pkl --input output/features_BANKNIFTY.json --symbol BANKNIFTY --trained-on NIFTY_500 --json-out frontend/frontend/public/data/regime_NIFTY500_on_BANKNIFTY.json

# 5. Deployment
Write-Host "`n--- Pushing Updates to GitHub ---" -ForegroundColor Cyan
git add .
git commit -m "Manual pipeline update: $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
git push

Write-Host "`nPipeline completed successfully! [DONE]" -ForegroundColor Green
