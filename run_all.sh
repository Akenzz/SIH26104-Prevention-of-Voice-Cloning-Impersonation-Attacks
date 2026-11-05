#!/bin/bash
# run_all.sh — Starts the Python Backend, optional Relay, and the React Frontend

echo "Starting Realtime Backend in the background..."
cd realtime-backend
if [ -z "$VIRTUAL_ENV" ] && [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
fi

python3 server.py &
BACKEND_PID=$!
cd ..

RELAY_PID=""
if [ "${WITH_RELAY:-0}" = "1" ]; then
    echo "Starting Relay Service on port 8001..."
    cd relay-service
    python3 server.py &
    RELAY_PID=$!
    cd ..
fi

cleanup() {
    echo -e "\nStopping background services..."
    kill $BACKEND_PID 2>/dev/null || true
    if [ -n "$RELAY_PID" ]; then
        kill $RELAY_PID 2>/dev/null || true
    fi
    echo "All processes stopped. Goodbye!"
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Starting React Frontend in the foreground..."
echo "=========================================================="
echo " SIH26104 Voice Integrity System is RUNNING"
echo " Backend is running on port 8000"
if [ -n "$RELAY_PID" ]; then
    echo " Relay is running on port 8001"
fi
echo " Press Ctrl+C at any time to stop everything."
echo "=========================================================="
cd voice-integrity-frontend
npm run dev

cleanup
