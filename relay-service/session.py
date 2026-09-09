"""Call Session management for Caller <-> Receiver <-> Backend Relay."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from fastapi import WebSocket

from config import RelaySettings
from audio.buffer import DropOldestQueue
from audio.crossfade import ChunkCrossfadeProcessor
from audio.pcm import float32_to_pcm, pcm_to_float32
from backend_client import RealtimeBackendClient
from converters.base import VoiceConverter
from converters.factory import create_converter

logger = logging.getLogger("relay_service.session")


class CallSession:
    """Manages an active call session connecting Caller, Receiver(s), and Backend."""

    def __init__(
        self,
        session_id: str,
        settings: RelaySettings,
        converter: VoiceConverter | None = None,
    ):
        self.session_id = session_id
        self.settings = settings

        # WebSocket connections
        self.caller_ws: WebSocket | None = None
        self.receivers: set[WebSocket] = set()

        # Audio conversion
        self.converter: VoiceConverter = converter or create_converter(
            settings.converter_type,
            pitch_shift_semitones=settings.pitch_shift_semitones,
            model_path=settings.rvc_model_path,
            index_path=settings.rvc_index_path,
            device=settings.rvc_device,
        )
        self.crossfade_processor = ChunkCrossfadeProcessor(
            sample_rate=settings.sample_rate,
            chunk_ms=settings.rvc_window_ms,
            context_ms=settings.rvc_context_ms,
            crossfade_ms=settings.rvc_crossfade_ms,
            soft_limit_max=settings.soft_limit_max_abs,
        )

        # Audio streaming queue with drop-oldest latency drift mitigation (Mitigation #2)
        self.caller_queue: DropOldestQueue[bytes] = DropOldestQueue(maxsize=settings.queue_maxsize)

        # State flags
        self.spoof_enabled: bool = False
        self.is_active: bool = False
        self.out_seq: int = 0
        self.total_frames_forwarded: int = 0
        self.total_bytes_forwarded: int = 0

        # Backend client
        self.backend_client = RealtimeBackendClient(
            backend_url=settings.backend_ws_url,
            sample_rate=settings.sample_rate,
            encoding=settings.encoding,
            channels=settings.channels,
            reconnect_delay=settings.reconnect_delay_sec,
            tcp_nodelay=settings.tcp_nodelay,
        )
        self.backend_client.add_score_callback(self._on_backend_score)
        self.backend_client.add_status_callback(self._on_backend_status)

        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the session worker and backend connection."""
        if self.is_active:
            return
        self.is_active = True
        await self.backend_client.start()
        self._worker_task = asyncio.create_task(self._process_audio_worker())
        logger.info("CallSession [%s] started", self.session_id)

    async def stop(self) -> None:
        """Tear down session."""
        self.is_active = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
        await self.backend_client.stop()
        self.crossfade_processor.reset()
        self.caller_queue.clear()
        logger.info("CallSession [%s] stopped", self.session_id)

    async def register_caller(self, ws: WebSocket) -> None:
        self.caller_ws = ws
        await self.broadcast_to_receivers({
            "type": "caller_status",
            "event": "caller_connected",
            "caller_connected": True,
            "session_id": self.session_id,
            "spoof_enabled": self.spoof_enabled,
        })

    async def unregister_caller(self, ws: WebSocket) -> None:
        if self.caller_ws == ws:
            self.caller_ws = None
            await self.broadcast_to_receivers({
                "type": "caller_status",
                "event": "caller_disconnected",
                "caller_connected": False,
                "session_id": self.session_id,
            })

    async def register_receiver(self, ws: WebSocket) -> None:
        self.receivers.add(ws)
        # Inform receiver of current call state
        await ws.send_json({
            "type": "call_status",
            "event": "call_status",
            "session_id": self.session_id,
            "caller_connected": self.caller_ws is not None,
            "spoof_enabled": self.spoof_enabled,
            "converter": self.converter.get_metadata(),
            "backend_connected": self.backend_client.is_connected,
        })

    async def unregister_receiver(self, ws: WebSocket) -> None:
        self.receivers.discard(ws)

    async def enqueue_audio(self, pcm_bytes: bytes) -> None:
        """Enqueue incoming PCM chunk from caller."""
        if not self.is_active or not pcm_bytes:
            return
        dropped = self.caller_queue.put_nowait(pcm_bytes)
        if dropped:
            logger.debug(
                "Session [%s]: dropped oldest frame to prevent latency drift (queue max=%d)",
                self.session_id,
                self.settings.queue_maxsize,
            )

    async def set_spoof(self, enabled: bool) -> None:
        """Toggle spoof state with fast transition mitigation (Mitigation #3)."""
        if self.spoof_enabled == enabled:
            return

        self.spoof_enabled = bool(enabled)
        logger.info(
            "Session [%s] spoof toggled -> %s",
            self.session_id,
            "ENABLED" if self.spoof_enabled else "DISABLED",
        )

        # 1. Reset crossfade processor and converter state
        self.crossfade_processor.reset()
        self.converter.reset()

        # 2. Fast transition mitigation: flush backend RingBuffer and EMA
        # Cuts transition detection latency from ~5.0s down to ~1.2s!
        await self.backend_client.flush_on_spoof_toggle()

        # 3. Broadcast transition state to receiver(s)
        await self.broadcast_to_receivers({
            "type": "transition_state",
            "status": "switching",
            "spoof_enabled": self.spoof_enabled,
            "timestamp": time.time(),
        })

        # 4. Confirm to caller
        if self.caller_ws:
            try:
                await self.caller_ws.send_json({
                    "type": "spoof_status",
                    "enabled": self.spoof_enabled,
                })
            except Exception:
                pass

    async def _process_audio_worker(self) -> None:
        """Continuous worker that pulls audio from caller queue, applies RVC/DSP
        if spoof is active, and sends to receiver(s) and backend.
        """
        try:
            while self.is_active:
                chunk_bytes = await self.caller_queue.get()
                if not chunk_bytes:
                    continue

                samples = pcm_to_float32(chunk_bytes, encoding=self.settings.encoding)
                if samples.size == 0:
                    continue

                if self.spoof_enabled:
                    # Transform through RVC / pitch-formant converter with crossfading & soft-limiting
                    converted_chunks = self.crossfade_processor.process(
                        samples,
                        self.converter.convert,
                    )
                    for c in converted_chunks:
                        out_pcm = float32_to_pcm(c, encoding=self.settings.encoding)
                        await self._forward_frame(out_pcm)
                else:
                    # Clean bonafide audio pass-through
                    out_pcm = float32_to_pcm(samples, encoding=self.settings.encoding)
                    await self._forward_frame(out_pcm)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Error in audio worker for session %s: %s", self.session_id, exc)

    async def _forward_frame(self, pcm_bytes: bytes) -> None:
        """Forward an audio frame to both the receiver(s) and the detection backend."""
        self.out_seq += 1
        self.total_frames_forwarded += 1
        self.total_bytes_forwarded += len(pcm_bytes)

        # 1. Forward binary frame to all connected receivers
        dead_receivers = set()
        for rx in self.receivers:
            try:
                await rx.send_bytes(pcm_bytes)
            except Exception:
                dead_receivers.add(rx)
        for dead in dead_receivers:
            self.receivers.discard(dead)

        # 2. Forward to detection backend
        await self.backend_client.send_audio(pcm_bytes)

    async def _on_backend_score(self, score_data: dict[str, Any]) -> None:
        """When backend scores a window, broadcast the score to receiver(s)."""
        await self.broadcast_to_receivers(score_data)

    async def _on_backend_status(self, status: str, details: dict[str, Any]) -> None:
        """Broadcast backend status changes to receiver(s)."""
        await self.broadcast_to_receivers({
            "type": "backend_status",
            "status": status,
            "details": details,
        })

    async def broadcast_to_receivers(self, message: dict[str, Any]) -> None:
        """Broadcast JSON message to all connected receiver sockets."""
        dead = set()
        for rx in self.receivers:
            try:
                await rx.send_json(message)
            except Exception:
                dead.add(rx)
        for d in dead:
            self.receivers.discard(d)


class SessionManager:
    """Manages active call sessions across the relay service."""

    def __init__(self, settings: RelaySettings):
        self.settings = settings
        self.sessions: dict[str, CallSession] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_session(self, session_id: str = "default") -> CallSession:
        async with self._lock:
            if session_id not in self.sessions:
                session = CallSession(session_id=session_id, settings=self.settings)
                await session.start()
                self.sessions[session_id] = session
            return self.sessions[session_id]

    async def close_session(self, session_id: str) -> None:
        async with self._lock:
            session = self.sessions.pop(session_id, None)
            if session:
                await session.stop()

    async def close_all(self) -> None:
        async with self._lock:
            for session in list(self.sessions.values()):
                await session.stop()
            self.sessions.clear()
