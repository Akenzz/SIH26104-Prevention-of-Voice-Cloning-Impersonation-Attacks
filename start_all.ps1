# Start BOTH the backend and the frontend (Windows / PowerShell).
#
#   .\start_all.ps1
#
# Opens the backend in its own window, waits for it to report healthy, then runs
# the frontend in this window. Ctrl+C stops the frontend; close the backend
# window (or use -Stop) to stop the backend.
#
#   .\start_all.ps1 -Stop        kill whatever is already on :8000 and :5173
#   .\start_all.ps1 -SkipInstall skip pip install / npm install (faster restarts)
#
# No environment variables needed. The first run downloads both checkpoints from
# Hugging Face (~400 MB) into realtime-backend/model_cache/ and can take a few
# minutes; later runs load from disk and only make one HEAD request to confirm
# the cached copy still matches the Hub.

param(
    [switch]$Stop,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$backendDir = Join-Path $root "realtime-backend"
$frontendDir = Join-Path $root "voice-integrity-frontend"

function Stop-Port($port) {
    # A stale server from an earlier session will hold the port and silently
    # serve OLD code, which looks exactly like "my changes did nothing".
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($procId in ($conns.OwningProcess | Select-Object -Unique)) {
        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "Stopping $($proc.ProcessName) (PID $procId) on port $port" -ForegroundColor Yellow
            Stop-Process -Id $procId -Force
        }
    }
}

Stop-Port 8000
Stop-Port 5173
if ($Stop) { Write-Host "Stopped." -ForegroundColor Green; exit 0 }

if (-not $SkipInstall) {
    Write-Host "Installing backend dependencies..." -ForegroundColor Cyan
    python -m pip install -q -r (Join-Path $backendDir "requirements.txt")
    if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
        Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
        Push-Location $frontendDir; npm install; Pop-Location
    }
}

Write-Host "Starting backend in a new window..." -ForegroundColor Cyan
Start-Process -FilePath "powershell" -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$backendDir'; python -m uvicorn server:app --host 0.0.0.0 --port 8000"
)

# First run downloads ~400 MB, so wait generously rather than failing fast.
Write-Host "Waiting for the backend to come up (first run downloads models)..." -NoNewline
$ready = $false
foreach ($i in 1..120) {
    Start-Sleep -Seconds 3
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5
        if ($h.status -eq "ok") { $ready = $true; break }
    } catch { Write-Host "." -NoNewline }
}
Write-Host ""

if (-not $ready) {
    Write-Host "Backend did not report healthy in 6 minutes." -ForegroundColor Red
    Write-Host "Check the backend window for the actual error, then rerun." -ForegroundColor Red
    exit 1
}

Write-Host "Backend ready." -ForegroundColor Green
Write-Host ("  experts        : " + ($h.experts -join ", "))
Write-Host ("  decision expert: " + $h.decision_expert)
foreach ($e in $h.expert_details) {
    Write-Host ("    " + $e.label + "  calibrator=" + $e.calibrator_version)
}
Write-Host ""
Write-Host "Backend  : http://localhost:8000/health"
Write-Host "Frontend : https://localhost:5173   (self-signed cert - click through the warning)"
Write-Host "Ctrl+C stops the frontend; close the backend window to stop the backend."
Write-Host ""

Set-Location $frontendDir
npm run dev

