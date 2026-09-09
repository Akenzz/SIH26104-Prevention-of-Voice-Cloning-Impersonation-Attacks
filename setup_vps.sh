#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SIH26104 Voice Integrity System - VPS Automated Deployment Script
# ==============================================================================

echo "=========================================================="
echo " Starting SIH26104 Voice Integrity Deployment"
echo "=========================================================="

# 1. Ensure Docker is installed
if ! command -v docker &> /dev/null; then
    echo "[+] Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo systemctl enable --now docker
    if [ -n "${SUDO_USER:-}" ]; then
        sudo usermod -aG docker "$SUDO_USER"
    elif [ -n "${USER:-}" ]; then
        sudo usermod -aG docker "$USER" || true
    fi
    echo "[✓] Docker installed successfully."
else
    echo "[✓] Docker is already installed."
fi

# 2. Ensure Docker Compose is available
if ! docker compose version &> /dev/null; then
    echo "[+] Installing Docker Compose plugin..."
    sudo apt-get update && sudo apt-get install -y docker-compose-plugin
fi

# 3. Pull latest git changes if in git repo
if [ -d ".git" ]; then
    echo "[+] Pulling latest repository changes..."
    git pull origin main || true
fi

# 4. Build and start containers
echo "[+] Building and starting containers with Docker Compose..."
docker compose up -d --build

# 5. Wait for containers to become healthy
echo "[+] Waiting for services to initialize (downloading models on first run takes ~2 minutes)..."
for i in {1..30}; do
    if curl -s -f http://localhost:8000/health &> /dev/null; then
        echo "[✓] Realtime Detection Backend is HEALTHY (:8000)"
        break
    fi
    echo -n "."
    sleep 5
done
echo ""

# 6. Detect public IP
PUBLIC_IP=$(curl -s https://api.ipify.org || curl -s https://ifconfig.me || echo "<YOUR_VPS_IP>")

echo "=========================================================="
echo " 🎉 DEPLOYMENT COMPLETE & RUNNING!"
echo "=========================================================="
echo " Detection API Base URL : http://${PUBLIC_IP}:8000"
echo " Detection Healthcheck  : http://${PUBLIC_IP}:8000/health"
echo " Detection WebSocket    : ws://${PUBLIC_IP}:8000/ws"
echo ""
echo " Relay Service Base URL : http://${PUBLIC_IP}:8001"
echo " Relay Healthcheck      : http://${PUBLIC_IP}:8001/health"
echo " Relay Caller WebSocket : ws://${PUBLIC_IP}:8001/ws/caller"
echo " Relay Receiver WS      : ws://${PUBLIC_IP}:8001/ws/receiver"
echo ""
echo " Web Dashboard Frontend : http://${PUBLIC_IP}:5173"
echo "=========================================================="
echo " To monitor logs : docker compose logs -f"
echo " To stop servers : docker compose down"
echo "=========================================================="
