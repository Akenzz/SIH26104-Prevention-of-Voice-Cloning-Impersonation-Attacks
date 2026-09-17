"""FastAPI process: REST /health plus WebSocket /ws."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import warnings
warnings.filterwarnings("ignore")
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf

from audio.resample import to_mono, to_target_rate
from audio.ring_buffer import RingBuffer
from calibration import (
    assert_calibrator_gates_match,
    assert_single_expert_calibrator,
    load_calibrator,
    load_expert_calibrators,
)
from config import EXPERT_CALIBRATORS, EXPERT_LABELS, load_settings
from experts.loader import load_experts, prefetch_hub_files
from fusion import load_fusion
from narration import sanitize, stream_groq
from audio.vad import has_speech
from aggregation import LogitEMA
from pipeline import ConnectionState, process_single_window
from policy import band, decide, load_policy


def summarize_file_results(results: list[dict[str, Any]], experts: list[str], *, policy: Any) -> dict[str, Any]:
    """Aggregate a file's per-window results into the same overall probability the
    offline optimizer uses: a mean of the weighted per-window spoof probabilities,
    not the final window snapshot.
    """
    valid = [r for r in results if isinstance(r, dict) and r.get("weighted_probability") is not None]
    if not valid:
        return {
            "event": "summary",
            "overall_risk_state": "unavailable",
            "final_smoothed_probability": None,
            "weighted_spoof_probability": None,
            "spoof_windows_count": 0,
            "total_windows_count": 0,
            "valid_windows_scored": 0,
            "agreement": "N/A",
            "confidence_level": "low",
            "peak_time_sec": None,
            "experts": {name: {"name": name, "label": EXPERT_LABELS.get(name, name), "probability": None, "risk_state": "unavailable"} for name in experts},
        }

    # Prefer the RAW (un-smoothed) per-window probability so the file's overall
    # verdict is independent of segment order. Fall back to the displayed weighted
    # probability for records (e.g. older callers) that don't carry a raw value.
    def _verdict_p(r: dict[str, Any]) -> float:
        v = r.get("raw_probability")
        return float(v) if v is not None else float(r["weighted_probability"])

    weighted_values = [_verdict_p(r) for r in valid]
    overall_probability = float(np.mean(weighted_values))
    expert_probs: dict[str, list[float]] = {name: [] for name in experts}
    for r in valid:
        per_expert = r.get("raw_per_expert_probability") or r.get("per_expert_probability") or {}
        for name in experts:
            if name in per_expert:
                expert_probs[name].append(float(per_expert[name]))

    expert_details = {}
    for name in experts:
        values = expert_probs.get(name, [])
        prob = float(np.mean(values)) if values else None
        expert_details[name] = {
            "name": name,
            "label": EXPERT_LABELS.get(name, name),
            "probability": prob,
            "risk_state": band(prob, policy) if prob is not None else "unavailable",
        }

    expert_mean_probs = {name: float(np.mean(values)) for name, values in expert_probs.items() if values}
    if expert_mean_probs:
        high_count = sum(1 for p in expert_mean_probs.values() if p > 0.65)
        low_count = sum(1 for p in expert_mean_probs.values() if p < 0.35)
        if high_count == len(expert_mean_probs):
            agreement = "unanimous_spoof"
        elif low_count == len(expert_mean_probs):
            agreement = "unanimous_bonafide"
        elif high_count > low_count:
            agreement = "majority_spoof"
        elif low_count > high_count:
            agreement = "majority_bonafide"
        else:
            agreement = "mixed"
    else:
        agreement = "N/A"

    if overall_probability >= 0.75 or overall_probability <= 0.25:
        confidence_level = "high"
    elif overall_probability >= 0.55 or overall_probability <= 0.45:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    peak_index = max(range(len(valid)), key=lambda i: _verdict_p(valid[i]))
    peak_time_sec = valid[peak_index].get("start_time_sec")

    return {
        "event": "summary",
        "overall_risk_state": band(overall_probability, policy),
        "final_smoothed_probability": overall_probability,
        "weighted_spoof_probability": overall_probability,
        "spoof_windows_count": int(sum(1 for p in weighted_values if p > 0.5)),
        "total_windows_count": len(valid),
        "valid_windows_scored": len(valid),
        "agreement": agreement,
        "confidence_level": confidence_level,
        "peak_time_sec": peak_time_sec,
        "experts": expert_details,
    }

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("realtime_backend.calibration").setLevel(logging.WARNING)
logging.getLogger("realtime_backend.experts").setLevel(logging.WARNING)
logging.getLogger("realtime_backend.pipeline").setLevel(logging.WARNING)
logging.getLogger("fairseq").setLevel(logging.WARNING)
logger = logging.getLogger("realtime_backend")
logger.setLevel(logging.WARNING) # Suppress default backend logger as well

settings = load_settings()
experts: dict[str, Any] = {}
fusion = load_fusion(settings.fusion_path)
calibrator = load_calibrator(settings.calibrator_path)
policy = load_policy(settings.policy_path)
# One calibrator per expert. The decision band still comes from `calibrator`
# (the SINGLE_EXPERT one); these exist so each model reports its own honest
# probability instead of being read on a foreign logit scale.
expert_calibrators = load_expert_calibrators(EXPERT_CALIBRATORS, settings.experts, calibrator)
if "hybrid_maxbr" in expert_calibrators and "hybrid" not in expert_calibrators:
    expert_calibrators["hybrid"] = expert_calibrators["hybrid_maxbr"]
elif "hybrid" in expert_calibrators and "hybrid_maxbr" not in expert_calibrators:
    expert_calibrators["hybrid_maxbr"] = expert_calibrators["hybrid"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global experts
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    if settings.prefetch_models:
        prefetch_hub_files(settings.model_cache_dir, settings.experts)
    print("Starting to load backend models... (this takes ~2-3 mins wait till you see 'you can now use the Frontend!')")
    experts = load_experts(settings)
    for key in experts:
        print(f"loaded model {key}")
        
    assert_calibrator_gates_match(EXPERT_CALIBRATORS, experts)
    assert_single_expert_calibrator(
        settings.fusion_mode, settings.single_expert,
        settings.calibrator_path, EXPERT_CALIBRATORS,
    )
    print("The 2 models loaded, server started!")
    print("you can now use the Frontend!")
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
    names = list(experts) or settings.experts
    return JSONResponse(
        {
            "status": "ok",
            "experts": names,
            "expert_details": [
                {
                    "name": n,
                    "label": EXPERT_LABELS.get(n, n),
                    "calibrator_version": getattr(expert_calibrators.get(n), "version", None),
                    "is_decision_expert": n == settings.single_expert,
                }
                for n in names
            ],
            "decision_expert": settings.single_expert,
            "decision_label": {
                "lr_fusion": "LR Fusion (WavLM + LFCC)",
                "heuristic_avg": "25% WavLM + 75% LFCC-LCNN Hybrid",
                "heuristic": "WavLM + LFCC Heuristic",
                "single": EXPERT_LABELS.get(settings.single_expert, settings.single_expert),
            }.get(settings.fusion_mode, settings.fusion_mode),
            "fusion_mode": settings.fusion_mode,
            "window_sec": settings.window_sec,
            "hop_sec": settings.hop_sec,
            "target_sample_rate": settings.target_sample_rate,
            "threshold_version": policy.version,
            "calibrator_version": calibrator.version,
            "fusion_version": fusion.version,
            # Task F: whether the optional LLM narration path is live this run.
            "narration": {
                "enabled": settings.narration_enabled,
                "model": settings.groq_model,
            },
        }
    )

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    """Convenience endpoint for manual testing (entire file, multi-window, with smoothing)."""
    import torch
    import torchaudio
    import torchaudio.transforms as T

    try:
        contents = await file.read()
        audio, sr = sf.read(io.BytesIO(contents), dtype="float32")
    except Exception as e:
        logger.warning(f"Failed to load uploaded file: {e}")
        return JSONResponse({"error": "Failed to decode audio"}, status_code=400)

    try:
        # 1. Resample to 16kHz mono using the exact same torchaudio pipeline as evaluate_slices.py
        # sf.read returns [frames, channels], to_mono makes it [frames]
        channels = 1 if audio.ndim == 1 else audio.shape[1]
        mono = to_mono(audio, channels)
        
        # Convert to torch tensor [1, frames] to match offline pipeline
        audio_tensor = torch.from_numpy(mono).unsqueeze(0)
        if sr != settings.target_sample_rate:
            audio_tensor = T.Resample(sr, settings.target_sample_rate)(audio_tensor)
            
        resampled = audio_tensor.squeeze(0).numpy()

        # 2. Extract ALL windows
        window_samples = settings.window_samples
        hop_samples = settings.hop_samples
        
        if len(resampled) < window_samples:
            resampled = np.pad(resampled, (0, window_samples - len(resampled)))
            
        buffer = RingBuffer(window_samples, hop_samples)
        windows = buffer.push(resampled)
    except Exception as e:
        logger.exception("Failed to preprocess audio: %s", e)
        return JSONResponse({"error": f"Audio preprocessing failed: {e}"}, status_code=500)

    # Capture the set of canonical expert names before the generator runs so
    # alias keys injected by pipeline.py (e.g. "hybrid" for "hybrid_maxbr")
    # don't cause KeyError when used as dict keys in expert_logits.
    canonical_expert_names = set(experts.keys())
    
    async def event_generator():
        try:
            # Per-expert LOGIT history for the file. The DISPLAYED per-window curve
            # uses a bounded rolling mean over the last ~2 s of window scores (not a
            # cumulative mean over the whole clip), so a spoof prefix stops dragging
            # every later bonafide window down only ~1/N per step. The overall verdict
            # is computed separately from the RAW per-window probabilities (see
            # summarize_file_results), so it is independent of segment order
            # (spoof-then-bonafide reads the same as bonafide-then-spoof).
            # Initialise with canonical names AND known aliases so that
            # pipeline.py's alias injection never causes a KeyError.
            expert_logits: dict[str, list[float]] = {name: [] for name in canonical_expert_names}
            if "hybrid_maxbr" in canonical_expert_names:
                expert_logits.setdefault("hybrid", [])
            if "hybrid" in canonical_expert_names:
                expert_logits.setdefault("hybrid_maxbr", [])

            # Rolling window length measured in *window scores*: ~2 s of hops (>=1).
            rolling_windows = max(1, round(2.0 / settings.hop_sec))

            # Probability-space fusion, identical to optimize_weights.py. Applied to
            # both the rolling (displayed) probs and the raw (verdict) probs so the
            # two paths differ only in their smoothing, not in their fusion.
            W_WAVLM, W_LFCC = 0.25, 0.75

            def fuse_probs(prob_map: dict[str, float]) -> float:
                p_w = prob_map.get("wavlm", 0.0)
                p_l = prob_map.get("hybrid_maxbr", prob_map.get("hybrid", 0.0))
                has_lfcc = "hybrid_maxbr" in prob_map or "hybrid" in prob_map
                present = ("wavlm" in prob_map) * W_WAVLM + has_lfcc * W_LFCC
                if present > 0:
                    fused = float((p_w * W_WAVLM + p_l * W_LFCC) / present)
                else:
                    fused = float((p_w + p_l) / 2.0)
                # Agreement override: both experts strongly agree spoof in THIS
                # window. Computed per-window, so unlike the old cumulative path it
                # is never "sticky" across the rest of the file.
                if p_l > 0.85 and p_w > 0.60:
                    fused = max(fused, 0.80)
                return fused

            results: list[dict] = []
            spoof_windows: list[int] = []
            valid_windows_scored = 0
            skipped_windows = 0
            
            for i, window in enumerate(windows):
                window_index = i + 1
                start_time_sec = i * settings.hop_sec
                
                # VAD Filtering
                is_speech = await asyncio.to_thread(has_speech, window, settings.target_sample_rate)
                if not is_speech:
                    skipped_windows += 1
                    yield f"data: {json.dumps({'event': 'skipped', 'window_index': window_index, 'start_time_sec': start_time_sec})}\n\n"
                    continue

                valid_windows_scored += 1
                
                # Score window (run in thread to prevent blocking the async SSE generator)
                result = await asyncio.to_thread(
                    process_single_window,
                    window,
                    settings,
                    experts,
                    fusion,
                    calibrator,
                    policy,
                    valid_windows_scored,
                    None,
                    False,
                    expert_calibrators
                )
                
                state, flag = result.risk_state, result.audio_quality
                scores, quality = result.scores, result.quality
                
                if not quality.ok:
                    yield f"data: {json.dumps({'event': 'quality_fail', 'window_index': window_index, 'reason': quality.reason})}\n\n"
                    continue
                    
                # Logit aggregation & fusion. For each expert we keep both:
                #   * raw_expert_probs      -- this window's own logit (feeds the verdict)
                #   * smoothed_expert_probs -- rolling mean of the last ~2 s (for display)
                raw_expert_probs = {}
                smoothed_expert_probs = {}
                for name, score_data in scores.items():
                    if name not in expert_logits:
                        expert_logits[name] = []
                    raw_logit = float(score_data["logit"])
                    expert_logits[name].append(raw_logit)
                    tail = expert_logits[name][-rolling_windows:]
                    rolling_logit = sum(tail) / len(tail)

                    cal = expert_calibrators.get(name, calibrator)
                    raw_expert_probs[name] = float(cal.probability(raw_logit))
                    smoothed_expert_probs[name] = float(cal.probability(rolling_logit))

                # Ensure both hybrid aliases are available in each map.
                for pm in (raw_expert_probs, smoothed_expert_probs):
                    if "hybrid_maxbr" in pm and "hybrid" not in pm:
                        pm["hybrid"] = pm["hybrid_maxbr"]
                    elif "hybrid" in pm and "hybrid_maxbr" not in pm:
                        pm["hybrid_maxbr"] = pm["hybrid"]

                # Displayed per-window probability (rolling ~2 s) and the
                # order-independent raw probability that feeds the file verdict.
                probability = fuse_probs(smoothed_expert_probs)
                raw_probability = fuse_probs(raw_expert_probs)

                state, action, flag = decide(
                    quality_ok=True,
                    quality_reason=None,
                    windows_scored=valid_windows_scored,
                    smoothed_probability=probability,
                    config=policy,
                )
                
                # Count spoof windows from the RAW (un-smoothed) probability so the
                # count reflects the actual spoof segments regardless of their order.
                if raw_probability > 0.5:
                    spoof_windows.append(window_index)

                # Only include canonical expert keys in the payload to avoid
                # confusing the frontend with alias duplicates.
                raw_scores_payload = {}
                prob_payload = {}
                raw_prob_payload = {}
                for name in canonical_expert_names:
                    if name in scores:
                        raw_scores_payload[name] = float(scores[name]["logit"])
                    if name in smoothed_expert_probs:
                        prob_payload[name] = smoothed_expert_probs[name]
                    if name in raw_expert_probs:
                        raw_prob_payload[name] = raw_expert_probs[name]

                window_payload = {
                    "event": "window_scored",
                    "window_index": window_index,
                    "start_time_sec": start_time_sec,
                    "risk_state": state,
                    # Displayed curve = rolling ~2 s (what the UI plots).
                    "calibrated_probability": probability,
                    "weighted_probability": probability,
                    "heuristic_avg_probability": probability,
                    # Raw per-window values = order-independent inputs to the verdict.
                    "raw_probability": raw_probability,
                    "raw_per_expert_scores": raw_scores_payload,
                    "per_expert_probability": prob_payload,
                    "raw_per_expert_probability": raw_prob_payload,
                }
                results.append(window_payload)
                yield f"data: {json.dumps(window_payload)}\n\n"
                
                # Yield control to the event loop to flush the TCP buffer
                await asyncio.sleep(0.01)
                
            # Final summary
            if valid_windows_scored == 0:
                summary = {
                    "event": "summary",
                    "overall_risk_state": "unavailable",
                    "final_smoothed_probability": None,
                    "weighted_spoof_probability": None,
                    "spoof_windows_count": 0,
                    "total_windows_count": len(windows),
                    "skipped_windows_count": skipped_windows,
                    "experts": []
                }
            else:
                summary = summarize_file_results(results, list(experts), policy=policy)
                summary["spoof_windows_count"] = len(spoof_windows)
                summary["total_windows_count"] = len(windows)
                summary["skipped_windows_count"] = skipped_windows
                summary["decision_expert"] = settings.single_expert
                expert_summary = summary.get("experts", {})
                summary["experts"] = [
                    {
                        "name": name,
                        "label": EXPERT_LABELS.get(name, name),
                        "probability": expert_summary.get(name, {}).get("probability"),
                        "risk_state": expert_summary.get(name, {}).get("risk_state", "unavailable"),
                    }
                    for name in canonical_expert_names
                ]
            yield f"data: {json.dumps(summary)}\n\n"
        except Exception as exc:
            logger.exception("Error during predict-file streaming: %s", exc)
            yield f"data: {json.dumps({'event': 'error', 'detail': str(exc)})}\n\n"
        
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )



@app.post("/narrate")
async def narrate(request: Request):
    """Task F: stream a one-line LLM narration of one window's numbers as SSE.

    Returns 503 when no GROQ_API_KEY is configured — the frontend then uses its
    local template narrator. On any upstream error mid-stream we emit an `error`
    SSE event and close, and the frontend falls back to local for that line.
    Only the allowlisted fields (see narration.sanitize) are ever forwarded.
    """
    if not settings.narration_enabled:
        return JSONResponse(
            {"detail": "narration disabled: no GROQ_API_KEY"}, status_code=503
        )
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"detail": "invalid JSON"}, status_code=400)
    fields = sanitize(payload if isinstance(payload, dict) else {})

    async def event_stream():
        try:
            async for delta in stream_groq(
                fields, api_key=settings.groq_api_key, model=settings.groq_model
            ):
                yield f"data: {json.dumps({'delta': delta})}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:  # network / upstream / auth / missing httpx
            logger.warning("narration upstream failed: %s", exc)
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
        expert_calibrators=expert_calibrators,
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
                except (ValueError, ImportError) as exc:
                    # ImportError: the streaming resampler needs scipy whenever
                    # the client's rate differs from the target. Report it on the
                    # socket instead of dropping the connection with a 1011.
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

    uvicorn.run("server:app", host=settings.host, port=settings.port, reload=False, log_level="warning")
