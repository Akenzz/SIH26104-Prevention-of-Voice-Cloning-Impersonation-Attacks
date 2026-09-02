#!/bin/bash
# run_all.sh — Starts both the Python Backend and the React Frontend

echo "Starting Realtime Backend in the background..."
cd realtime-backend
if [ -z "$VIRTUAL_ENV" ]; then
    # Activate virtual environment if not already active
    source ../.venv/bin/activate
fi

# Start backend in the background
python3 server.py &
BACKEND_PID=$!
cd ..

# Trap Ctrl+C (SIGINT) to clean up the backend when exiting
cleanup() {
    echo -e "\nStopping Backend (PID: $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null || true
    echo "All processes stopped. Goodbye!"
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Starting React Frontend in the foreground..."
echo "=========================================================="
echo " SIH26104 Voice Integrity System is RUNNING"
echo " Backend is running on port 8000"
echo " Press Ctrl+C at any time to stop everything."
echo "=========================================================="
cd voice-integrity-frontend
npm run dev

# If npm run dev exits naturally, clean up backend anyway
cleanup
