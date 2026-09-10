# VPS Container Deployment Guide

This guide walks you through deploying the **SIH26104 Realtime Voice Integrity Detection Backend** and **Relay Service** on your VPS using Docker and Docker Compose.

---

## 1. Quick Architecture Overview

When deployed with `docker compose up -d --build`, two containers run in an isolated bridge network:

```
[ Mobile App / Flutter / Vercel Web ]
       │                      │
       │ (Live Calls)         │ (Direct File / Audio Stream)
       ▼                      ▼
┌──────────────────┐   ┌────────────────────────┐
│  relay-service   │   │    realtime-backend    │
│   (Port 8001)    │──▶│      (Port 8000)       │
│                  │   │  WavLM + LFCC Experts  │
└──────────────────┘   └────────────────────────┘
                                   │
                           [Persistent Volume]
                          (realtime_model_cache)
```

- **`realtime-backend` (Port 8000):** FastAPI server running WavLM Base+ and LFCC-LCNN models for deepfake detection (`/health`, `/predict-file`, `/ws`).
- **`relay-service` (Port 8001):** High-concurrency call relay server handling caller/receiver streams and voice spoofing simulation (`/ws/caller`, `/ws/receiver`, `/health`).
- **Persistent Volume:** Model weights (~400MB) are downloaded only once into Docker volume `realtime_model_cache`. Future restarts take under 5 seconds!

---

## 2. One-Command Setup on Your VPS

### Step 1: Install Docker & Docker Compose (If Not Already Installed)
SSH into your VPS and run:
```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Enable & start Docker service
sudo systemctl enable --now docker

# (Optional) Allow current user to run Docker without sudo
sudo usermod -aG docker $USER
newgrp docker
```

### Step 2: Clone the Repository & Launch Containers
```bash
# Clone the repository
git clone https://github.com/Akenzz/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks.git
cd SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks

# Build and start the containers in the background
docker compose up -d --build
```

---

## 3. Verify Container Status & Health

Check running containers:
```bash
docker compose ps
```
You should see:
```
NAME                    IMAGE               COMMAND                  SERVICE             STATUS
voice_realtime_backend  ...                 "python server.py"       realtime-backend    Up (healthy)
voice_relay_service     ...                 "python server.py"       relay-service       Up (healthy)
```

Verify the health endpoints:
```bash
# Check Detection Backend
curl http://localhost:8000/health

# Check Relay Service
curl http://localhost:8001/health
```

To follow live logs:
```bash
docker compose logs -f
```

---

## 4. What "Base URL" to Give Your Teammate

Your teammate asked for the **Base URL** to configure the frontend (which they are deploying on Vercel):

### For Detection / Dashboard:
- **HTTP Base URL:** `http://<YOUR_VPS_IP>:8000`
- **WebSocket URL:** `ws://<YOUR_VPS_IP>:8000/ws`

### For Live Calling / Flutter App:
- **Relay HTTP Base URL:** `http://<YOUR_VPS_IP>:8001`
- **Caller WebSocket:** `ws://<YOUR_VPS_IP>:8001/ws/caller?session_id=call-1&client_role=caller`
- **Receiver WebSocket:** `ws://<YOUR_VPS_IP>:8001/ws/receiver?session_id=call-1&client_role=receiver`

---

## 5. Important: Handling HTTPS / WSS for Vercel Frontend

> [!WARNING]
> If your teammate hosts the frontend on Vercel (`https://something.vercel.app`), modern browsers (Chrome, Safari, Edge) will **block** plain `http://` or `ws://` calls due to **Mixed Content Security Restrictions**.
> 
> *Note: Native mobile apps (Flutter / Android APK) are **not** affected by this browser restriction and can connect directly to `ws://<YOUR_VPS_IP>:8001`!*

If the teammate's web app on Vercel needs to connect to the backend, you have two quick options to get free SSL (`https://` and `wss://`):

### Option A: Free Cloudflare Tunnel (Zero Domain / Zero SSL Config)
Run Cloudflare Tunnel directly on your VPS:
```bash
# Download cloudflared on your VPS
curl -L --output cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared
sudo mv cloudflared /usr/local/bin/

# Expose Detection Backend (port 8000):
cloudflared tunnel --url http://localhost:8000
```
Cloudflare will print a public HTTPS URL (e.g., `https://random-name.trycloudflare.com`). Give that URL to your teammate!

### Option B: Reverse Proxy with Caddy (If You Have a Domain Name)
If you have a domain or subdomain pointing to your VPS IP (e.g. `api.yourdomain.com`):
1. Install Caddy:
   ```bash
   sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
   sudo apt update && sudo apt install -y caddy
   ```
2. Put this in `/etc/caddy/Caddyfile`:
   ```caddy
   api.yourdomain.com {
       reverse_proxy localhost:8000
   }

   relay.yourdomain.com {
       reverse_proxy localhost:8001
   }
   ```
3. Run `sudo systemctl restart caddy`. Caddy will automatically generate and renew Let's Encrypt SSL certificates!

---

## 6. Maintenance Commands

- **Stop containers:**
  ```bash
  docker compose down
  ```
- **Update with latest code:**
  ```bash
  git pull
  docker compose up -d --build
  ```
- **View resource usage:**
  ```bash
  docker stats
  ```
- **Clean unused docker data:**
  ```bash
  docker system prune -f
  ```
