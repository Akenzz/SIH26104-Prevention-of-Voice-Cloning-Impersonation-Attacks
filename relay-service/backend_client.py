"""WebSocket client maintaining connection to the realtime detection backend."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from typing import Any, Callable, Coroutine
import websockets

logger = logging.getLogger("relay_service.backend_client")

ScoreCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None] | None]
StatusCallback = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None] | None]


class RealtimeBackendClient:
    """Manages the connection to the detection backend's /ws endpoint.
    
    Adheres strictly to the protocol:
      1. Handshake: sends start message declaring pcm_s16le, 16kHz, mono, binary_seq=false
      2. Waits for {"type": "ready"} before streaming audio
      3. Sets TCP_NODELAY to avoid latency accumulation
      4. Fast transition: flushes backend RingBuffer/EMA on spoof toggle by re-issuing start
      5. Normalizes and forwards detection score messages
    """

    def __init__(
        self,
        backend_url: str = "ws://127.0.0.1:8000/ws",
        sample_rate: int = 16000,
        encoding: str = "pcm_s16le",
        channels: int = 1,
        reconnect_delay: float = 1.0,
        tcp_nodelay: bool = True,
    ):
        self.backend_url = backend_url
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.reconnect_delay = reconnect_delay
        self.tcp_nodelay = tcp_nodelay

        self._ws: Any = None
        self._is_ready: bool = False
        self._is_running: bool = False
        self._reader_task: asyncio.Task | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._score_callbacks: list[ScoreCallback] = []
        self._status_callbacks: list[StatusCallback] = []
        self._ready_event = asyncio.Event()

    @property
    def is_connected(self) -> bool:
        if self._ws is None:
            return False
        state = getattr(self._ws, "state", None)
        if state is not None:
            return getattr(state, "name", "") == "OPEN"
        return getattr(self._ws, "open", False) or not getattr(self._ws, "closed", True)

    @property
    def is_ready(self) -> bool:
        return self.is_connected and self._is_ready

    def add_score_callback(self, cb: ScoreCallback) -> None:
        self._score_callbacks.append(cb)

    def add_status_callback(self, cb: StatusCallback) -> None:
        self._status_callbacks.append(cb)

    async def start(self) -> None:
        """Start backend client and initiate connection."""
        if self._is_running:
            return
        self._is_running = True
        await self._connect()

    async def stop(self) -> None:
        """Close connection and background tasks."""
        self._is_running = False
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._is_ready = False
        self._ready_event.clear()

    async def _connect(self) -> bool:
        """Connect to backend, set TCP_NODELAY, and execute handshake."""
        self._is_ready = False
        self._ready_event.clear()
        try:
            logger.info("Connecting to detection backend at %s", self.backend_url)
            self._ws = await websockets.connect(
                self.backend_url,
                max_size=None,
                ping_interval=20,
                ping_timeout=20,
            )

            # Apply TCP_NODELAY to avoid TCP buffer bloat (Mitigation #2)
            if self.tcp_nodelay:
                self._apply_tcp_nodelay(self._ws)

            # Perform handshake (Mitigation #1)
            start_payload = {
                "type": "start",
                "sample_rate": self.sample_rate,
                "encoding": self.encoding,
                "channels": self.channels,
                "binary_seq": False,
            }
            await self._ws.send(json.dumps(start_payload))

            # Await ready response
            raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
            ready_msg = json.loads(raw)
            if ready_msg.get("type") != "ready":
                raise RuntimeError(f"Expected 'ready' response from backend, got: {ready_msg}")

            self._is_ready = True
            self._ready_event.set()
            logger.info("Backend handshake completed successfully: %s", ready_msg)
            await self._notify_status("connected", ready_msg)

            # Start background reader task
            if self._reader_task and not self._reader_task.done():
                self._reader_task.cancel()
            self._reader_task = asyncio.create_task(self._reader_loop())
            return True

        except Exception as exc:
            logger.warning("Backend connection failed: %s", exc)
            self._is_ready = False
            self._ready_event.clear()
            await self._notify_status("disconnected", {"error": str(exc)})
            self._schedule_reconnect()
            return False

    def _schedule_reconnect(self) -> None:
        if self._is_running and (self._reconnect_task is None or self._reconnect_task.done()):
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        while self._is_running and not self.is_ready:
            await asyncio.sleep(self.reconnect_delay)
            if not self._is_running:
                break
            success = await self._connect()
            if success:
                break

    def _apply_tcp_nodelay(self, ws: Any) -> None:
        try:
            # Under websockets >= 10, transport is in ws.transport or ws.protocol.transport
            transport = getattr(ws, "transport", None) or getattr(getattr(ws, "protocol", None), "transport", None)
            if transport:
                sock = transport.get_extra_info("socket")
                if sock:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    logger.debug("TCP_NODELAY enabled on backend connection")
        except Exception as exc:
            logger.debug("Could not set TCP_NODELAY on backend connection: %s", exc)

    async def _reader_loop(self) -> None:
        """Reads JSON score frames from backend and dispatches to callbacks."""
        try:
            while self._is_running and self.is_connected:
                msg_raw = await self._ws.recv()
                if isinstance(msg_raw, bytes):
                    continue
                try:
                    payload = json.loads(msg_raw)
                except json.JSONDecodeError:
                    continue

                # If backend sends ready (e.g. after flush)
                if payload.get("type") == "ready":
                    self._is_ready = True
                    self._ready_event.set()
                    await self._notify_status("ready", payload)
                    continue

                # Normalize score payload
                normalized = self._normalize_score_event(payload)
                await self._notify_score(normalized)

        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed as exc:
            logger.warning("Backend connection closed: %s", exc)
        except Exception as exc:
            logger.exception("Error in backend reader loop: %s", exc)
        finally:
            self._is_ready = False
            self._ready_event.clear()
            await self._notify_status("disconnected", {})
            self._schedule_reconnect()

    def _normalize_score_event(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize backend response to provide consistent fields:
        risk_state, calibrated_probability, weighted_probability, per_expert_probability.
        """
        # Score message can have type == 'score' or event == 'window_scored'
        risk_state = raw.get("risk_state", "unavailable")
        prob = raw.get("smoothed_probability")
        if prob is None:
            prob = raw.get("calibrated_probability")
        if prob is None:
            prob = raw.get("weighted_probability")

        # Extract per-expert probabilities
        per_expert_prob = raw.get("per_expert_probability") or {}
        if not per_expert_prob and "scores" in raw and isinstance(raw["scores"], dict):
            for name, score_data in raw["scores"].items():
                if isinstance(score_data, dict) and "probability" in score_data:
                    per_expert_prob[name] = score_data["probability"]

        # Ensure both event and type are present
        normalized = dict(raw)
        normalized["event"] = "window_scored"
        normalized["type"] = "score"
        normalized["risk_state"] = risk_state
        normalized["calibrated_probability"] = prob
        normalized["weighted_probability"] = prob
        normalized["smoothed_probability"] = prob
        normalized["per_expert_probability"] = per_expert_prob

        return normalized

    async def send_audio(self, pcm_bytes: bytes) -> bool:
        """Send raw binary PCM bytes to the backend."""
        if not self.is_ready or self._ws is None:
            return False
        try:
            await self._ws.send(pcm_bytes)
            return True
        except Exception as exc:
            logger.warning("Failed to send audio to backend: %s", exc)
            return False

    async def flush_on_spoof_toggle(self) -> None:
        """Fast transition mitigation (Mitigation #3):
        Re-send the JSON start handshake to the backend.
        This triggers ConnectionState.apply_start() in the backend, flushing
        its RingBuffer, resampler filter state, and EMA smoothing trackers.
        Cuts transition detection latency from ~5s down to ~1.2s!
        """
        if not self.is_connected or self._ws is None:
            return

        logger.info("Fast transition triggered: flushing backend RingBuffer & EMA")
        start_payload = {
            "type": "start",
            "sample_rate": self.sample_rate,
            "encoding": self.encoding,
            "channels": self.channels,
            "binary_seq": False,
        }
        try:
            await self._ws.send(json.dumps(start_payload))
        except Exception as exc:
            logger.warning("Failed to send flush start to backend: %s", exc)

    async def _notify_score(self, score_data: dict[str, Any]) -> None:
        for cb in self._score_callbacks:
            try:
                res = cb(score_data)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as exc:
                logger.debug("Error in score callback: %s", exc)

    async def _notify_status(self, status: str, details: dict[str, Any]) -> None:
        for cb in self._status_callbacks:
            try:
                res = cb(status, details)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as exc:
                logger.debug("Error in status callback: %s", exc)
