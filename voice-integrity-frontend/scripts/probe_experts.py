"""Patient WebSocket probe: proves which experts actually score each window.

The backend's own `scripts/wav_client.py` drains the socket with 50-200 ms
timeouts, which is fine for LFCC-LCNN but too impatient for WavLM on CPU
(one 4 s window costs ~1-3 s there), so it exits with "No score messages
received" even though the server scored fine. This client sends the whole clip
first, then waits as long as it takes and prints the per-expert logits from
`raw_per_expert_scores` so you can see Role A (wavlm) and Role D (lfcc) side by
side.

Usage (from voice-integrity-frontend/, backend already running):

    python scripts/probe_experts.py --wav test-samples/sweep_6s.wav
    python scripts/probe_experts.py --wav some_LA_clip.flac --drain 120

Requires: numpy, websockets (already needed by the backend).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave
from pathlib import Path

import numpy as np


def load_audio(path: Path) -> tuple[np.ndarray, int, int]:
    """float32 mono-or-multichannel samples + sample rate + channel count."""
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wf:
            channels, rate, width = wf.getnchannels(), wf.getframerate(), wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        if width == 2:
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            samples = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
        else:
            raise SystemExit(f"unsupported sample width: {width} bytes")
        return samples, rate, channels

    try:
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit(f"reading {path.suffix} needs soundfile: pip install soundfile") from exc
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    return np.asarray(audio, dtype=np.float32).reshape(-1), int(rate), channels


def fmt_experts(scores) -> str:
    """raw_per_expert_scores comes back as a dict or a list of records."""
    if not scores:
        return "(none reported)"
    if isinstance(scores, dict):
        items = scores.items()
    else:
        items = [
            (r.get("expert") or r.get("name") or "?", r.get("score", r.get("logit")))
            for r in scores
        ]
    return "  ".join(
        f"{name}={value:+.4f}" if isinstance(value, (int, float)) else f"{name}={value}"
        for name, value in items
    )


async def run(url: str, wav: Path, chunk_ms: int, drain_s: float) -> int:
    import websockets

    samples, rate, channels = load_audio(wav)
    print(f"clip: {wav}  rate={rate}Hz  channels={channels}  dur={samples.size / rate / channels:.1f}s")

    pcm = np.asarray(samples, dtype="<f4").tobytes()
    frame_bytes = 4 * channels
    chunk_bytes = max(frame_bytes, int(rate * chunk_ms / 1000.0) * frame_bytes)

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "start",
                    "sample_rate": rate,
                    "encoding": "pcm_f32le",
                    "channels": channels,
                }
            )
        )
        ready = json.loads(await ws.recv())
        print(f"ready: experts={ready.get('experts')}  fusion={ready.get('fusion_mode')}")

        for offset in range(0, len(pcm), chunk_bytes):
            await ws.send(pcm[offset : offset + chunk_bytes])
        print(f"sent {len(pcm)} bytes; draining up to {drain_s:.0f}s (WavLM on CPU is slow)…")

        seen = 0
        while True:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=drain_s))
            except asyncio.TimeoutError:
                break
            except websockets.exceptions.ConnectionClosed:
                break
            if msg.get("type") not in (None, "score"):
                print(f"  [{msg.get('type')}] {json.dumps(msg)[:200]}")
                continue
            seen += 1
            p = msg.get("smoothed_probability")
            print(
                f"  seq={msg.get('sequence_number'):>3}  state={msg.get('risk_state'):<11} "
                f"p={'  n/a ' if p is None else f'{p:.4f}'}  "
                f"latency={msg.get('latency_ms')}ms  quality={msg.get('audio_quality')}"
            )
            print(f"        experts: {fmt_experts(msg.get('raw_per_expert_scores'))}")
            if seen == 1:
                print(f"        full first message: {json.dumps(msg)}")

    print(f"\n{seen} scored window(s).")
    return 0 if seen else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--drain", type=float, default=60.0, help="seconds to wait per message")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args.url, args.wav, args.chunk_ms, args.drain)))


if __name__ == "__main__":
    main()
