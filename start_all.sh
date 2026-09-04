#!/usr/bin/env bash
# Start BOTH the backend and the frontend (Linux / macOS).
#
#   ./start_all.sh                 start both
#   ./start_all.sh --stop          kill whatever is on :8000 and :5173
#   ./start_all.sh --skip-install  skip pip/npm install (faster restarts)
#
# The backend runs in the background with its log in /tmp; the frontend runs in
# the foreground. Ctrl+C stops both.
#
# No environment variables needed. The first run downloads both checkpoints from
# Hugging Face (~400 MB) into realtime-backend/model_cache/ and can take a few
# minutes; later runs load from disk.

set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
backend_log="/tmp/sih-backend.log"
skip_install=0
stop_only=0
for a in "$@"; do
  case "$a" in
    --stop) stop_only=1 ;;
    --skip-install) skip_install=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

stop_port() {
  # A stale server from an earlier run holds the port and silently serves OLD
  # code, which looks exactly like "my changes did nothing".
  local pids
  pids="$(lsof -ti ":$1" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "Stopping PID(s) $pids on port $1"
    kill -9 $pids 2>/dev/null || true
  fi
}

stop_port 8000
stop_port 5173
[ "$stop_only" -eq 1 ] && { echo "Stopped."; exit 0; }

if [ "$skip_install" -eq 0 ]; then
  echo "Installing backend dependencies..."
  python3 -m pip install -q -r "$root/realtime-backend/requirements.txt"
  [ -d "$root/voice-integrity-frontend/node_modules" ] || \
    (echo "Installing frontend dependencies..."; cd "$root/voice-integrity-frontend" && npm install)
fi

echo "Starting backend (log: $backend_log)..."
( cd "$root/realtime-backend" && python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 ) \
  >"$backend_log" 2>&1 &
backend_pid=$!
trap 'kill $backend_pid 2>/dev/null || true' EXIT INT TERM

printf 'Waiting for the backend (first run downloads models)'
ready=0
for _ in $(seq 1 120); do
  sleep 3
  if curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$backend_pid" 2>/dev/null || { echo; echo "Backend exited. Last lines:"; tail -20 "$backend_log"; exit 1; }
  printf '.'
done
echo

if [ "$ready" -eq 0 ]; then
  echo "Backend did not report healthy in 6 minutes. Last lines of $backend_log:" >&2
  tail -20 "$backend_log" >&2
  exit 1
fi

python3 - <<'PY'
import json, urllib.request
h = json.load(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5))
print("Backend ready.")
print("  experts        :", ", ".join(h["experts"]))
print("  decision expert:", h.get("decision_expert"))
for e in h.get("expert_details", []):
    print(f"    {e['label']}  calibrator={e['calibrator_version']}")
PY

echo
echo "Backend  : http://localhost:8000/health"
echo "Frontend : https://localhost:5173   (self-signed cert - click through the warning)"
echo "Ctrl+C stops both."
echo

cd "$root/voice-integrity-frontend"
npm run dev
