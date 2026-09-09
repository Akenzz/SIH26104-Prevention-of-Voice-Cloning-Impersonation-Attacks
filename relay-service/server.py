"""FastAPI application for the Relay Service: /ws/caller, /ws/receiver, and /health."""

from __future__ import annotations

import base64
import json
import logging
import socket
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import load_settings
from session import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("relay_service")

settings = load_settings()
session_manager = SessionManager(settings)


def apply_tcp_nodelay(ws: WebSocket) -> None:
    """Set TCP_NODELAY on the underlying socket to prevent buffer bloat and latency drift."""
    try:
        transport = ws.scope.get("transport")
        if transport:
            sock = transport.get_extra_info("socket")
            if sock:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception as exc:
        logger.debug("Could not apply TCP_NODELAY to WebSocket: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Relay Service starting on %s:%d", settings.host, settings.port)
    logger.info("Backend WS target: %s", settings.backend_ws_url)
    logger.info("Active voice converter type: %s", settings.converter_type)
    yield
    logger.info("Relay Service shutting down, closing active sessions...")
    await session_manager.close_all()


app = FastAPI(title="Voice Integrity Relay Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health status and summary of active relay sessions."""
    default_session = session_manager.sessions.get("default")
    backend_ok = default_session.backend_client.is_connected if default_session else False

    return JSONResponse({
        "status": "ok",
        "active_sessions": len(session_manager.sessions),
        "backend_url": settings.backend_ws_url,
        "default_backend_connected": backend_ok,
        "sample_rate": settings.sample_rate,
        "encoding": settings.encoding,
        "converter_type": settings.converter_type,
        "experts": ["wavlm", "hybrid", "ssl"],
        "expert_details": [
            {
                "name": "wavlm",
                "label": "Expert 1 · WavLM Base+",
                "calibrator_version": "wavlm-base-plus-ep6",
                "is_decision_expert": False,
            },
            {
                "name": "hybrid",
                "label": "Expert 2 · LFCC-LCNN Hybrid",
                "calibrator_version": "hybrid_clean_plus_newclips_final",
                "is_decision_expert": False,
            },
            {
                "name": "ssl",
                "label": "Expert 3 · TakHemlata SSL",
                "calibrator_version": "best_SSL_model_LA",
                "is_decision_expert": True,
            },
        ],
        "decision_expert": "ssl",
        "decision_label": "LR Fusion (WavLM + LFCC + SSL)",
        "fusion_mode": "lr_fusion",
        "window_sec": 4.0,
        "hop_sec": 0.5,
        "target_sample_rate": settings.sample_rate,
    })


class SpoofToggleRequest(BaseModel):
    enabled: bool


@app.post("/api/session/{session_id}/spoof")
async def api_toggle_spoof(session_id: str, body: SpoofToggleRequest):
    """REST API endpoint to toggle spoof on an active call session."""
    session = await session_manager.get_or_create_session(session_id)
    await session.set_spoof(body.enabled)
    return {
        "session_id": session_id,
        "spoof_enabled": session.spoof_enabled,
    }


@app.get("/api/session/{session_id}/status")
async def api_session_status(session_id: str):
    """Retrieve session details and stats."""
    if session_id not in session_manager.sessions:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    session = session_manager.sessions[session_id]
    return {
        "session_id": session_id,
        "caller_connected": session.caller_ws is not None,
        "receiver_count": len(session.receivers),
        "spoof_enabled": session.spoof_enabled,
        "total_frames_forwarded": session.total_frames_forwarded,
        "total_bytes_forwarded": session.total_bytes_forwarded,
        "backend_connected": session.backend_client.is_connected,
        "converter": session.converter.get_metadata(),
    }


@app.websocket("/ws/caller")
async def websocket_caller(
    ws: WebSocket,
    session_id: str = Query(default="default"),
):
    """Caller phone / client WebSocket endpoint.
    
    Receives:
      - Binary PCM chunks: raw PCM audio bytes (16-bit signed LE mono 16kHz)
      - JSON: {"type": "start", ...}, {"type": "toggle_spoof", "enabled": bool},
              {"type": "audio", "data": "base64..."}
    """
    await ws.accept()
    apply_tcp_nodelay(ws)
    session = await session_manager.get_or_create_session(session_id)
    await session.register_caller(ws)
    logger.info("Caller connected to session [%s]", session_id)

    try:
        # Acknowledge connection
        await ws.send_json({
            "type": "ready",
            "session_id": session_id,
            "sample_rate": settings.sample_rate,
            "encoding": settings.encoding,
            "spoof_enabled": session.spoof_enabled,
        })

        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break

            # 1. Binary audio frame
            raw_bytes = message.get("bytes")
            if raw_bytes is not None:
                await session.enqueue_audio(raw_bytes)
                continue

            # 2. Text / JSON message
            text = message.get("text")
            if text is None:
                continue

            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = str(payload.get("type") or payload.get("event") or "").lower()

            if msg_type in {"toggle_spoof", "set_spoof"}:
                enabled = bool(payload.get("enabled", not session.spoof_enabled))
                await session.set_spoof(enabled)

            elif msg_type == "start":
                await ws.send_json({
                    "type": "ready",
                    "session_id": session_id,
                    "sample_rate": settings.sample_rate,
                    "encoding": settings.encoding,
                    "spoof_enabled": session.spoof_enabled,
                })

            elif msg_type in {"audio", "frame"}:
                data_b64 = payload.get("data") or payload.get("pcm") or payload.get("audio")
                if data_b64 and isinstance(data_b64, str):
                    try:
                        pcm_bytes = base64.b64decode(data_b64)
                        await session.enqueue_audio(pcm_bytes)
                    except Exception:
                        pass

            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

            elif msg_type == "set_profile":
                profile = payload.get("profile")
                if profile and hasattr(session.converter, "set_profile"):
                    session.converter.set_profile(profile)
                    await ws.send_json({
                        "type": "profile_updated",
                        "profile": profile,
                    })

    except WebSocketDisconnect:
        logger.info("Caller disconnected from session [%s]", session_id)
    except Exception as exc:
        logger.exception("Caller WebSocket crashed in session [%s]: %s", session_id, exc)
    finally:
        await session.unregister_caller(ws)


@app.websocket("/ws/receiver")
async def websocket_receiver(
    ws: WebSocket,
    session_id: str = Query(default="default"),
):
    """Receiver phone / client WebSocket endpoint.
    
    Streams:
      - Binary PCM chunks: forwarded audio (bonafide or spoof-converted)
      - JSON frames: window_scored events with risk_state, calibrated_probability,
                     weighted_probability, per_expert_probability.
      - JSON events: caller_status, transition_state, call_status.
      
    Accepts:
      - Control messages: {"type": "toggle_spoof", "enabled": bool}, {"type": "ping"}
    """
    await ws.accept()
    apply_tcp_nodelay(ws)
    session = await session_manager.get_or_create_session(session_id)
    await session.register_receiver(ws)
    logger.info("Receiver connected to session [%s]", session_id)

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break

            text = message.get("text")
            if text:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue

                msg_type = str(payload.get("type") or payload.get("event") or "").lower()
                if msg_type in {"toggle_spoof", "set_spoof"}:
                    enabled = bool(payload.get("enabled", not session.spoof_enabled))
                    await session.set_spoof(enabled)

                elif msg_type == "ping":
                    await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("Receiver disconnected from session [%s]", session_id)
    except Exception as exc:
        logger.exception("Receiver WebSocket crashed in session [%s]: %s", session_id, exc)
    finally:
        await session.unregister_receiver(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=settings.host, port=settings.port, reload=False, log_level="info")
