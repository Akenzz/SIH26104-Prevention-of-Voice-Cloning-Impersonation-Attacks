"""Diagnostic: replay an audio FILE through the EXACT live /ws pipeline.

This is not the /predict-file path. It drives pipeline.ConnectionState the same
way the WebSocket handler does (StreamingPolyphaseResampler + per-window EMA +
50/50 heuristic_avg fusion + policy), so the per-window numbers it prints are
what the Live Monitor would show for this clip -- minus the over-the-air loss of
actually playing it through a speaker into a mic.

Usage:
    python replay_live.py <audio_file> [--sr-in 48000]

--sr-in fakes the capture sample rate the browser would declare (the mic
AudioContext is usually 48 kHz); the clip is resampled to it first so the live
resampler runs on the same 48k->16k path the real stream uses. Omit it to feed
the clip at its native rate.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import soundfile as sf

from calibration import load_calibrator, load_expert_calibrators
from config import EXPERT_CALIBRATORS, load_settings
from experts.loader import load_experts
from fusion import load_fusion
from pipeline import ConnectionState
from policy import load_policy


def _resample_to(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    # Simple linear resample just to fake the capture rate; the pipeline's own
    # polyphase resampler does the real 48k->16k anti-aliased step afterwards.
    n_out = int(round(len(x) * sr_to / sr_from))
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    fp = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(fp, xp, x).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--sr-in", type=int, default=48000,
                    help="Fake capture rate to declare to the pipeline (default 48000, like a browser mic).")
    ap.add_argument("--chunk", type=int, default=2048,
                    help="Feed the clip in chunks of this many input samples, to mimic streaming.")
    args = ap.parse_args()

    settings = load_settings()
    print(f"experts={settings.experts} fusion={settings.fusion_mode} "
          f"single={settings.single_expert} ema_alpha={settings.ema_alpha}")
    fusion = load_fusion(settings.fusion_path)
    calibrator = load_calibrator(settings.calibrator_path)
    policy = load_policy(settings.policy_path)
    experts = load_experts(settings)
    expert_calibrators = load_expert_calibrators(EXPERT_CALIBRATORS, settings.experts, calibrator)
    if "hybrid_maxbr" in expert_calibrators and "hybrid" not in expert_calibrators:
        expert_calibrators["hybrid"] = expert_calibrators["hybrid_maxbr"]

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = _resample_to(audio, sr, args.sr_in)
    dur = len(audio) / args.sr_in
    print(f"loaded {args.audio}: native_sr={sr} fed_sr={args.sr_in} dur={dur:.2f}s samples={len(audio)}")

    state = ConnectionState(
        settings=settings, experts=experts, fusion=fusion,
        calibrator=calibrator, policy=policy, expert_calibrators=expert_calibrators,
    )
    state.apply_start({"sample_rate": args.sr_in, "encoding": "pcm_f32le", "channels": 1})

    print(f"{'win':>4} {'t(s)':>6} {'state':>11} {'smoothed':>9}  per-expert(prob)")
    peak = -1.0
    n = 0
    for off in range(0, len(audio), args.chunk):
        chunk = audio[off:off + args.chunk]
        for msg in state.ingest_json_frame({"type": "audio", "audio": chunk.tolist(),
                                             "sample_rate": args.sr_in, "channels": 1}):
            if msg.get("type") == "error":
                continue
            sm = msg.get("smoothed_probability")
            scores = msg.get("scores") or {}
            per = " ".join(
                f"{k}={ (v.get('probability') if isinstance(v, dict) else None) }"
                for k, v in scores.items() if k in ("wavlm", "hybrid_maxbr", "hybrid")
            )
            t = (msg.get("window_index") or 0) * settings.hop_sec
            smtxt = f"{sm:.3f}" if isinstance(sm, (int, float)) else str(sm)
            print(f"{msg.get('window_index'):>4} {t:>6.1f} {str(msg.get('risk_state')):>11} {smtxt:>9}  {per}")
            if isinstance(sm, (int, float)):
                peak = max(peak, sm)
            n += 1

    print(f"\nwindows_scored~{n}  peak_smoothed={peak:.3f}"
          if peak >= 0 else f"\nwindows_scored~{n}  peak_smoothed=n/a")


if __name__ == "__main__":
    sys.exit(main())
