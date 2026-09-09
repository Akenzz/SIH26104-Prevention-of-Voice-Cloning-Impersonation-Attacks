FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RELAY_HOST=0.0.0.0 \
    RELAY_PORT=8001 \
    BACKEND_WS_URL=ws://realtime-backend:8000/ws \
    CONVERTER_TYPE=mock

WORKDIR /app

# Install system dependencies needed for audio manipulation and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app

EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "server.py"]
