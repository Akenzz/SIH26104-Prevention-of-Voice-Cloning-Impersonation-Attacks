"""FastAPI process: REST /health plus WebSocket /ws."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from calibration import load_calibrator
from config import load_settings
from experts.loader import load_experts, prefetch_hub_files
from fusion import load_fusion
from pipeline import ConnectionState
from policy import load_policy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("realtime_backend")

settings = load_settings()
experts: dict[str, Any] = {}
fusion = load_fusion(settings.fusion_path)
calibrator = load_calibrator(settings.calibrator_path)
policy = load_policy(settings.policy_path)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global experts
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    if settings.prefetch_models:
        logger.info("PREFETCH_MODELS=1: downloading Hub checkpoints into %s", settings.model_cache_dir)
        prefetch_hub_files(settings.model_cache_dir)
    logger.info("Loading experts: %s (fusion_mode=%s)", settings.experts, settings.fusion_mode)
    experts = load_experts(settings)
    logger.info("Backend ready on experts=%s", list(experts))
    yield


app = FastAPI(title="SIH26104 realtime backend", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "experts": list(experts) or settings.experts,
            "fusion_mode": settings.fusion_mode,
            "window_sec": settings.window_sec,
            "hop_sec": settings.hop_sec,
            "target_sample_rate": settings.target_sample_rate,
            "threshold_version": policy.version,
            "calibrator_version": calibrator.version,
            "fusion_version": fusion.version,
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    state = ConnectionState(
        settings=settings,
        experts=experts,
        fusion=fusion,
        calibrator=calibrator,
        policy=policy,
    )
    logger.info("WebSocket connected")
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                outgoing = state.ingest_binary(message["bytes"])
                for item in outgoing:
                    await ws.send_json(item)
                continue
            text = message.get("text")
            if text is None:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                await ws.send_json(state._error("invalid JSON"))
                continue
            msg_type = str(payload.get("type") or payload.get("event") or "").lower()
            if msg_type in {"start", "hello", "config"} or (
                not state.started and "sample_rate" in payload
            ):
                try:
                    ready = state.apply_start(payload)
                except ValueError as exc:
                    await ws.send_json(state._error(str(exc)))
                    continue
                await ws.send_json(ready)
                continue
            if msg_type in {"audio", "frame"}:
                outgoing = state.ingest_json_frame(payload)
                for item in outgoing:
                    await ws.send_json(item)
                continue
            await ws.send_json(state._error(f"unknown message type {msg_type!r}"))
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected seq=%s", state.out_seq)
    except Exception:
        logger.exception("WebSocket handler crashed")
        try:
            await ws.close(code=1011)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=settings.host, port=settings.port, reload=False)
