"""Configuration settings for the Python Relay Service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass
class RelaySettings:
    # Server network settings
    host: str = os.environ.get("RELAY_HOST", "0.0.0.0")
    port: int = int(os.environ.get("RELAY_PORT", "8001"))

    # Backend detection service URL
    backend_ws_url: str = os.environ.get("BACKEND_WS_URL", "ws://127.0.0.1:8000/ws")

    # Audio format specifications (must match backend protocol)
    sample_rate: int = int(os.environ.get("SAMPLE_RATE", "16000"))
    encoding: str = os.environ.get("ENCODING", "pcm_s16le")
    channels: int = int(os.environ.get("CHANNELS", "1"))
    chunk_ms: int = int(os.environ.get("CHUNK_MS", "100"))

    # RVC / Converter processing window specifications
    rvc_window_ms: int = int(os.environ.get("RVC_WINDOW_MS", "200"))
    rvc_context_ms: int = int(os.environ.get("RVC_CONTEXT_MS", "150"))
    rvc_crossfade_ms: int = int(os.environ.get("RVC_CROSSFADE_MS", "25"))
    soft_limit_max_abs: float = float(os.environ.get("SOFT_LIMIT_MAX_ABS", "0.92"))

    # Latency drift mitigation
    queue_maxsize: int = int(os.environ.get("QUEUE_MAXSIZE", "20"))
    tcp_nodelay: bool = os.environ.get("TCP_NODELAY", "1") == "1"

    # Conversion engine
    converter_type: str = os.environ.get("CONVERTER_TYPE", "mock")
    pitch_shift_semitones: float = float(os.environ.get("PITCH_SHIFT", "4.0"))
    rvc_model_path: str = os.environ.get("RVC_MODEL_PATH", "")
    rvc_index_path: str = os.environ.get("RVC_INDEX_PATH", "")
    rvc_device: str = os.environ.get("RVC_DEVICE", "cpu")

    # Backend reconnect settings
    reconnect_delay_sec: float = float(os.environ.get("RECONNECT_DELAY_SEC", "1.0"))
    reconnect_max_attempts: int = int(os.environ.get("RECONNECT_MAX_ATTEMPTS", "10"))

    @property
    def bytes_per_sample(self) -> int:
        return 2 if "s16" in self.encoding or "int16" in self.encoding else 4

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * (self.chunk_ms / 1000.0))

    @property
    def chunk_bytes(self) -> int:
        return self.chunk_samples * self.bytes_per_sample * self.channels

    @property
    def rvc_window_samples(self) -> int:
        return int(self.sample_rate * (self.rvc_window_ms / 1000.0))

    @property
    def rvc_context_samples(self) -> int:
        return int(self.sample_rate * (self.rvc_context_ms / 1000.0))

    @property
    def rvc_crossfade_samples(self) -> int:
        return int(self.sample_rate * (self.rvc_crossfade_ms / 1000.0))


def load_settings() -> RelaySettings:
    return RelaySettings()
