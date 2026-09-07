"""Feed a WAV/FLAC (or generated tone) through the live WebSocket path.

Usage:
    python scripts/wav_client.py
    python scripts/wav_client.py --wav path/to/file.wav --url ws://127.0.0.1:8000/ws
    python scripts/wav_client.py --wav path/to/file.flac   # any soundfile-readable format

WAV is read with the stdlib ``wave`` module (no deps). Any other extension
(``.flac``, ``.ogg``, ...) is read via ``soundfile`` if it is installed, so LA
``.flac`` clips can be fed directly without converting them first.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_wav_float32(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        width = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
    if width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    else:
        raise ValueError(f"unsupported sample width {width}")
    return samples, rate, channels


def load_soundfile_float32(path: Path) -> tuple[np.ndarray, int, int]:
    """Read any non-WAV format (FLAC, OGG, ...) via soundfile, as float32."""
    try:
        import soundfile as sf
    except ImportError as exc:  # keep the WAV path dependency-free
        raise SystemExit(
            f"reading {path.suffix} needs soundfile: pip install soundfile"
        ) from exc
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    return np.asarray(audio, dtype=np.float32).reshape(-1), int(rate), channels


def load_audio_float32(path: Path) -> tuple[np.ndarray, int, int]:
    """Dispatch on extension: stdlib wave for .wav, soundfile for everything else."""
    if path.suffix.lower() == ".wav":
        return load_wav_float32(path)
    return load_soundfile_float32(path)


def make_tone(seconds: float = 8.0, rate: int = 16000) -> tuple[np.ndarray, int, int]:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False, dtype=np.float32)
    audio = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    audio += np.random.default_rng(0).normal(0, 0.02, audio.size).astype(np.float32)
    return audio, rate, 1


async def run(url: str, wav: Path | None, chunk_ms: int) -> None:
    import websockets

    if wav is None:
        samples, rate, channels = make_tone()
        print(f"No WAV given; sending {samples.size / rate:.1f}s generated tone at {rate} Hz")
    else:
        samples, rate, channels = load_audio_float32(wav)
        print(f"Loaded {wav} rate={rate} channels={channels} samples={samples.size}")

    pcm = np.asarray(samples, dtype="<f4").tobytes()
    bytes_per_sample = 4 * channels
    chunk_bytes = max(bytes_per_sample, int(rate * (chunk_ms / 1000.0) * bytes_per_sample))

    seqs: list[int] = []
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
        print("ready:", json.dumps(ready))
        offset = 0
        while offset < len(pcm):
            await ws.send(pcm[offset : offset + chunk_bytes])
            offset += chunk_bytes
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.05)
                    msg = json.loads(raw)
                    seqs.append(int(msg.get("sequence_number", -1)))
                    print(
                        f"seq={msg.get('sequence_number')} state={msg.get('risk_state')} "
                        f"p={msg.get('smoothed_probability')} latency_ms={msg.get('latency_ms')} "
                        f"quality={msg.get('audio_quality')}"
                    )
            except asyncio.TimeoutError:
                continue
        await asyncio.sleep(0.5)
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.2)
                msg = json.loads(raw)
                seqs.append(int(msg.get("sequence_number", -1)))
                print(
                    f"seq={msg.get('sequence_number')} state={msg.get('risk_state')} "
                    f"p={msg.get('smoothed_probability')} latency_ms={msg.get('latency_ms')}"
                )
        except asyncio.TimeoutError:
            pass

    if not seqs:
        raise SystemExit("No score messages received")
    if seqs != list(range(seqs[0], seqs[0] + len(seqs))):
        raise SystemExit(f"Sequence numbers not monotonic: {seqs}")
    print(f"OK: {len(seqs)} messages, sequence {seqs[0]}..{seqs[-1]}")


async def run_sse(url: str, wav: Path | None) -> None:
    import httpx

    if wav is None:
        raise SystemExit("SSE test requires an actual file path (--wav)")

    print(f"Testing SSE POST to {url} with {wav}")
    async with httpx.AsyncClient() as client:
        with open(wav, "rb") as f:
            files = {"file": (wav.name, f, "audio/wav")}
            async with client.stream("POST", url, files=files) as response:
                if response.status_code != 200:
                    print(f"Error {response.status_code}: {await response.aread()}")
                    return
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[len("data: "):]
                        msg = json.loads(data)
                        print(f"SSE Event: {msg.get('event')} | "
                              f"seq={msg.get('window_index', '-')} "
                              f"state={msg.get('risk_state', msg.get('overall_risk_state'))} "
                              f"p={msg.get('calibrated_probability', msg.get('final_smoothed_probability'))}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    parser.add_argument("--sse-url", default="http://127.0.0.1:8000/predict-file")
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--sse", action="store_true", help="Test the SSE /predict-file endpoint instead of websockets")
    args = parser.parse_args()
    
    if args.sse:
        asyncio.run(run_sse(args.sse_url, args.wav))
    else:
        asyncio.run(run(args.url, args.wav, args.chunk_ms))

if __name__ == "__main__":
    main()

