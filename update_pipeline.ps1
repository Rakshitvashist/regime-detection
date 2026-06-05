# update_pipeline.ps1
# ---------------------------------------------------------
# Manual script to update data, retrain models, and deploy.
#
# Pipeline per in-sample instrument:
#   Rust core (features)  ->  kmeans_regime (causal labels)
#     ->  xgb_regime (train + walk-forward OOF history for the dashboard)
#     ->  export_model_metrics (Model-tab diagnostics)
# Cross-instrument OOS predictions reuse the trained models on other series.
#
# Note: xgb_regime.py now writes the served frontend file itself using
# out-of-fold (honest) predictions, so the old in-sample predict.py calls are
# gone. predict.py remains a standalone in-sample/live inference utility.
# ---------------------------------------------------------

$Python = ".\.venv\Scripts\python.exe"
$Data = "frontend/frontend/public/data"

# Abort the whole run if any step fails, instead of committing/pushing
# half-updated or stale outputs (previously every step ran unconditionally and
# the script always reached git push, even after a failure).
$ErrorActionPreference = "Stop"

function Invoke-Step {
    param([string]$Label, [scriptblock]$Action)
    Write-Host "  -> $Label" -ForegroundColor DarkGray
    & $Action
    if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
        throw "Step failed (exit $LASTEXITCODE): $Label"
    }
}

Write-Host "--- Starting Regime Detection Update Pipeline ---" -ForegroundColor Cyan

# 0. Fetch Latest Data
Write-Host "`n[0/4] Fetching latest data from Yahoo Finance..." -ForegroundColor Yellow
Invoke-Step "fetch_data.py" { & $Python fetch_data.py }

# --- Helper: full per-symbol in-sample build (features -> labels -> train -> metrics) ---
function Build-Instrument($csv, $symbol, $start) {
    Write-Host "`nProcessing $symbol..." -ForegroundColor Yellow
    Invoke-Step "core features ($symbol)" { cargo run --manifest-path core/Cargo.toml --release -- "$csv" $symbol "$start" }
    Invoke-Step "kmeans labels ($symbol)" { & $Python research/kmeans_regime.py --input "output/features_$symbol.json" --output "output/regime_clustered_$symbol.json" --symbol $symbol }
    # --shift 0 => DETECTION (nowcast the current regime, ~90% accuracy). The old
    # 21-day-ahead direction forecast scored ~34% (direction is unforecastable);
    # forward-looking signal now lives in the risk ladder of forecast_horizons.py.
    Invoke-Step "xgb train ($symbol)" { & $Python research/xgb_regime.py --input "output/features_$symbol.json" --model-out "output/xgb_model_$symbol.pkl" --labels "output/regime_clustered_$symbol.json" --symbol $symbol --shift 0 }
    Invoke-Step "model metrics ($symbol)" { & $Python research/export_model_metrics.py --model "output/xgb_model_$symbol.pkl" --input "output/features_$symbol.json" --symbol $symbol --json-out "$Data/metrics_$symbol.json" --predictions "$Data/regime_$symbol.json" }
    # Multi-horizon forward forecaster (5/10/15/21d) + deploy gate -> forecast_<symbol>.json
    Invoke-Step "horizon forecast ($symbol)" { & $Python research/forecast_horizons.py --input "output/features_$symbol.json" --symbol $symbol --json-out "$Data/forecast_$symbol.json" }
}

# 1-3. In-sample instruments
Build-Instrument "NSE_NIFTY, 1D.csv"     "NIFTY"     "2019-01-01"
Build-Instrument "NSE_BANKNIFTY, 1D.csv" "BANKNIFTY" "2019-01-01"
Build-Instrument "NSE_CNX500, 1D.csv"    "NIFTY_500" "2019-01-01"

# 4. Cross-Asset OOS Experiments (Oil & Forex) - features only, no labels
Write-Host "`n[4/4] Processing Cross-Asset OOS (Oil & Forex)..." -ForegroundColor Yellow
Invoke-Step "core features (CRUDE)"  { cargo run --manifest-path core/Cargo.toml --release -- "MCX_CRUDEOIL1!, 1D.csv" CRUDE  "2019-01-01" }
Invoke-Step "core features (WTI)"    { cargo run --manifest-path core/Cargo.toml --release -- "CFI_WTI, 1D.csv"        WTI    "2019-01-01" }
Invoke-Step "core features (USDINR)" { cargo run --manifest-path core/Cargo.toml --release -- "FX_IDC_USDINR, 1D.csv"  USDINR "2019-01-01" }

# predict_oos.py: trained-model -> target-series. Each tuple is (model, input, symbol, trainedOn, out).
$OOS = @(
    @("NIFTY",     "CRUDE",     "regime_NIFTY_on_CRUDE.json"),
    @("NIFTY",     "WTI",       "regime_NIFTY_on_WTI.json"),
    @("NIFTY",     "USDINR",    "regime_NIFTY_on_USDINR.json"),
    @("NIFTY_500", "CRUDE",     "regime_NIFTY500_on_CRUDE.json"),
    @("NIFTY_500", "WTI",       "regime_NIFTY500_on_WTI.json"),
    @("NIFTY_500", "USDINR",    "regime_NIFTY500_on_USDINR.json"),
    @("BANKNIFTY", "NIFTY",     "regime_BANKNIFTY_on_NIFTY.json"),
    @("BANKNIFTY", "NIFTY_500", "regime_BANKNIFTY_on_NIFTY500.json"),
    @("NIFTY_500", "NIFTY",     "regime_NIFTY500_on_NIFTY.json"),
    @("NIFTY_500", "BANKNIFTY", "regime_NIFTY500_on_BANKNIFTY.json"),
    @("NIFTY",     "NIFTY_500", "regime_NIFTY_on_NIFTY500.json"),
    @("NIFTY",     "BANKNIFTY", "regime_NIFTY_on_BANKNIFTY.json")
)
foreach ($o in $OOS) {
    $trainedOn = $o[0]; $target = $o[1]; $out = $o[2]
    Invoke-Step "OOS $trainedOn -> $target" {
        & $Python research/predict_oos.py --model "output/xgb_model_$trainedOn.pkl" --input "output/features_$target.json" --symbol $target --trained-on $trainedOn --json-out "$Data/$out" --horizon 0
    }
}

# 5. Deployment - only commit/push when every step above succeeded AND there are
# actually changes to commit (a no-op commit otherwise aborts under -Stop).
Write-Host "`n--- Pushing Updates to GitHub ---" -ForegroundColor Cyan
git add -A
$pending = git status --porcelain
if ([string]::IsNullOrWhiteSpace($pending)) {
    Write-Host "No changes to commit - skipping push." -ForegroundColor Yellow
} else {
    git commit -m "Manual pipeline update: $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
    git push --set-upstream origin main
    if ($LASTEXITCODE -ne 0) { throw "git push failed" }
}

Write-Host "`nPipeline completed successfully! [DONE]" -ForegroundColor Green
