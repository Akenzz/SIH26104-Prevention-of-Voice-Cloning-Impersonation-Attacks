"""FastAPI process: REST /health plus WebSocket /ws."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import io
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf

from audio.resample import to_mono, to_target_rate
from calibration import load_calibrator
from config import load_settings
from experts.loader import load_experts, prefetch_hub_files
from fusion import load_fusion
from pipeline import ConnectionState, process_single_window
from policy import load_policy
from runtime_contract import validate_runtime_contract

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
    # Verify model/artifact compatibility before downloads or model startup so a
    # configuration error cannot accidentally produce plausible-looking scores.
    validate_runtime_contract(settings, calibrator, fusion)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    if settings.prefetch_models:
        logger.info("PREFETCH_MODELS=1: downloading Hub checkpoints into %s", settings.model_cache_dir)
        prefetch_hub_files(settings.model_cache_dir)
    logger.info("Loading experts: %s (fusion_mode=%s)", settings.experts, settings.fusion_mode)
    experts = load_experts(settings)
    logger.info("Backend ready on experts=%s", list(experts))
    yield


app = FastAPI(title="SIH26104 realtime backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> JSONResponse:
    loaded_experts = list(experts) or settings.experts
    is_dummy = any(name == "dummy" for name in loaded_experts)
    return JSONResponse(
        {
            "status": "degraded" if is_dummy else "ok",
            "experts": loaded_experts,
            "fusion_mode": settings.fusion_mode,
            "window_sec": settings.window_sec,
            "hop_sec": settings.hop_sec,
            "target_sample_rate": settings.target_sample_rate,
            "threshold_version": policy.version,
            "calibrator_version": calibrator.version,
            "fusion_version": fusion.version,
        }
    )

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    """Convenience endpoint for manual testing (entire file, multi-window, with smoothing)."""
    try:
        contents = await file.read()
        audio, sr = sf.read(io.BytesIO(contents), dtype="float32")
    except Exception as e:
        logger.warning(f"Failed to load uploaded file: {e}")
        # Return unavailable state on bad decode, triggering assess_window failure with empty array
        state, action, flag, scores, fused, probability, quality = process_single_window(
            window=np.zeros(0, dtype=np.float32), 
            settings=settings, experts=experts, fusion=fusion, 
            calibrator=calibrator, policy=policy, windows_scored=1, 
            smoothed_probability=None, dropped_frames=False
        )
        return JSONResponse({
            "summary": {
                "overall_risk_state": state,
                "final_smoothed_probability": None,
                "max_probability": None,
                "max_probability_window_index": None
            },
            "windows": [],
            "audio_quality": quality.reason
        })

    # 1. Resample to 16kHz mono (reuse logic)
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    mono = to_mono(audio, channels)
    resampled, did_resample = to_target_rate(mono, sr, settings.target_sample_rate)

    # 2. Extract ALL windows using RingBuffer (pad if shorter than 1 window)
    from audio.ring_buffer import RingBuffer
    from smoothing import ExponentialMovingAverage
    
    window_samples = settings.window_samples
    hop_samples = settings.hop_samples
    
    if len(resampled) < window_samples:
        resampled = np.pad(resampled, (0, window_samples - len(resampled)))
        
    buffer = RingBuffer(window_samples, hop_samples)
    windows = buffer.push(resampled)
    
    if len(windows) > 150: # ~10 mins
        logger.warning(f"Uploaded file has {len(windows)} windows; processing may take a while.")

    # 3. Score all windows and apply EMA
    ema = ExponentialMovingAverage(settings.ema_alpha)
    expert_emas = {name: ExponentialMovingAverage(settings.ema_alpha) for name in experts}
    
    results = []
    max_prob = -1.0
    max_prob_idx = -1
    final_state = "unavailable"
    final_prob = None
    final_scores = {}
    final_quality_flag = None
    
    for i, window in enumerate(windows):
        window_index = i + 1
        start_time_sec = i * settings.hop_sec
        
        state, action, flag, scores, fused, probability, quality = process_single_window(
            window=window,
            settings=settings,
            experts=experts,
            fusion=fusion,
            calibrator=calibrator,
            policy=policy,
            windows_scored=window_index,
            smoothed_probability=None, # pass None first to get raw probability
            dropped_frames=False
        )
        
        if quality.ok and probability is not None:
            smoothed = ema.update(probability)
            # Re-run policy decision with the smoothed probability
            from policy import decide
            state, action, flag = decide(
                quality_ok=True,
                quality_reason=None,
                windows_scored=window_index,
                smoothed_probability=smoothed,
                config=policy,
            )
            final_prob = smoothed
            
            if smoothed > max_prob:
                max_prob = smoothed
                max_prob_idx = window_index
                
            # Update individual expert EMAs
            for name, score_dict in scores.items():
                expert_prob = calibrator.probability(score_dict["logit"])
                expert_emas[name].update(expert_prob)
        else:
            smoothed = None
            
        final_state = state
        final_scores = scores
        if flag or not quality.ok:
            final_quality_flag = flag if flag else quality.reason
            
        results.append({
            "window_index": window_index,
            "start_time_sec": start_time_sec,
            "risk_state": state,
            "calibrated_probability": smoothed if smoothed is not None else probability,
            "raw_per_expert_scores": {k: v["logit"] for k, v in scores.items()} if scores else {}
        })
        
    # Calculate final individual expert risk states
    expert_risk_states = {}
    from policy import decide
    for name, e_ema in expert_emas.items():
        if e_ema.value is not None:
            st, _, _ = decide(
                quality_ok=True,
                quality_reason=None,
                windows_scored=len(windows),
                smoothed_probability=e_ema.value,
                config=policy,
            )
            expert_risk_states[name] = st
        else:
            expert_risk_states[name] = "unavailable"
        
    summary = {
        "overall_risk_state": final_state,
        "final_smoothed_probability": final_prob,
        "max_probability": max_prob if max_prob >= 0 else None,
        "max_probability_window_index": max_prob_idx if max_prob_idx >= 0 else None,
        "expert_risk_states": expert_risk_states
    }
    
    return JSONResponse({
        "summary": summary,
        "windows": results,
        "model_version": {k: v["model_version"] for k, v in final_scores.items()} if final_scores else {},
        "calibrator_version": calibrator.version,
        "threshold_version": policy.version,
        "fusion_version": fusion.version,
        "audio_quality": final_quality_flag
    })


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
