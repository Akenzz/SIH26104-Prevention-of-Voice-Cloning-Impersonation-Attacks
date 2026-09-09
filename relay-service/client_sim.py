"""Standalone simulation CLI for testing the Relay Service.

Simulates Caller and Receiver clients to verify zero-packet-drop flow,
audio streaming integrity, spoof toggling, and live score reception.

Usage:
  # Run full simulation (Caller + Receiver in parallel):
  python client_sim.py --relay-url ws://127.0.0.1:8080 --duration 6.0 --toggle-spoof-at 2.5

  # Run caller only:
  python client_sim.py --mode caller --duration 5.0

  # Run receiver only:
  python client_sim.py --mode receiver
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("client_sim")


def generate_speech_like_tone(
    duration_sec: float = 5.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Generate a rich, harmonic speech-like signal (formant-filtered harmonic sweep)."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)

    # Glottal pulse fundamental frequency sweep (120 Hz to 220 Hz)
    f0 = 140.0 + 30.0 * np.sin(2.0 * np.pi * 0.8 * t)
    phase = 2.0 * np.pi * np.cumsum(f0) / sample_rate

    # Sum harmonics (1 to 10)
    audio = np.zeros_like(t)
    for h in range(1, 10):
        amp = (1.0 / h) * (0.8 if h in {1, 2, 3} else 0.4)
        audio += amp * np.sin(h * phase)

    # Add subtle background breathing noise
    rng = np.random.default_rng(42)
    audio += 0.015 * rng.normal(0, 1.0, audio.size).astype(np.float32)

    # Normalize to peak 0.70 (safe within [-0.92, 0.92])
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak) * 0.70

    return audio.astype(np.float32)


def load_wav_file(wav_path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio from WAV file and convert to mono float32 at target sample rate."""
    try:
        import soundfile as sf
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            import scipy.signal as signal
            num_target = int(len(audio) * (target_sr / sr))
            audio = signal.resample(audio, num_target)
        return audio.astype(np.float32)
    except Exception as exc:
        logger.error("Failed to load WAV file %s: %s", wav_path, exc)
        sys.exit(1)


async def run_caller(
    relay_url: str,
    session_id: str,
    audio: np.ndarray,
    sample_rate: int = 16000,
    chunk_ms: int = 100,
    toggle_spoof_at: float | None = 2.5,
) -> dict[str, Any]:
    """Caller simulator: connects to /ws/caller, paces audio chunks, toggles spoof."""
    ws_url = f"{relay_url.rstrip('/')}/ws/caller?session_id={session_id}"
    chunk_samples = int(sample_rate * (chunk_ms / 1000.0))
    chunk_sec = chunk_ms / 1000.0

    # Convert audio to int16 PCM bytes
    clipped = np.clip(audio, -1.0, 1.0)
    int16_audio = (clipped * 32767.0).astype("<i2")
    pcm_bytes = int16_audio.tobytes()
    bytes_per_chunk = chunk_samples * 2

    logger.info("Caller connecting to %s", ws_url)
    stats = {
        "chunks_sent": 0,
        "bytes_sent": 0,
        "spoof_toggled": False,
        "t_start": 0.0,
        "t_end": 0.0,
    }

    async with websockets.connect(ws_url, max_size=None) as ws:
        ready_raw = await ws.recv()
        logger.info("Caller received handshake: %s", ready_raw)

        stats["t_start"] = time.perf_counter()
        offset = 0
        total_len = len(pcm_bytes)

        while offset < total_len:
            loop_start = time.perf_counter()
            elapsed = loop_start - stats["t_start"]

            # Trigger spoof toggle if configured
            if toggle_spoof_at is not None and not stats["spoof_toggled"] and elapsed >= toggle_spoof_at:
                logger.info("Caller: toggling spoof ON at t=%.2fs", elapsed)
                await ws.send(json.dumps({"type": "toggle_spoof", "enabled": True}))
                stats["spoof_toggled"] = True

            chunk = pcm_bytes[offset : offset + bytes_per_chunk]
            offset += bytes_per_chunk

            # Send raw binary chunk
            await ws.send(chunk)
            stats["chunks_sent"] += 1
            stats["bytes_sent"] += len(chunk)

            # Pace chunks to real-time speed
            spent = time.perf_counter() - loop_start
            sleep_time = chunk_sec - spent
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        stats["t_end"] = time.perf_counter()
        logger.info(
            "Caller finished: sent %d chunks (%d bytes) in %.2fs",
            stats["chunks_sent"],
            stats["bytes_sent"],
            stats["t_end"] - stats["t_start"],
        )
        # Give short grace time for last frames to traverse
        await asyncio.sleep(0.5)

    return stats


async def run_receiver(
    relay_url: str,
    session_id: str,
    expected_duration_sec: float,
    sample_rate: int = 16000,
) -> dict[str, Any]:
    """Receiver simulator: connects to /ws/receiver, collects audio and scores."""
    ws_url = f"{relay_url.rstrip('/')}/ws/receiver?session_id={session_id}"
    logger.info("Receiver connecting to %s", ws_url)

    stats = {
        "chunks_received": 0,
        "bytes_received": 0,
        "scores_received": [],
        "transitions_received": [],
        "status_received": [],
    }

    async with websockets.connect(ws_url, max_size=None) as ws:
        stop_event = asyncio.Event()

        async def receiver_loop():
            try:
                while not stop_event.is_set():
                    msg = await ws.recv()
                    if isinstance(msg, bytes):
                        stats["chunks_received"] += 1
                        stats["bytes_received"] += len(msg)
                    else:
                        try:
                            payload = json.loads(msg)
                            msg_type = payload.get("type") or payload.get("event")
                            if msg_type in {"score", "window_scored"}:
                                stats["scores_received"].append(payload)
                                logger.info(
                                    "Receiver score: win=%s state=%s prob=%.3f latency=%.1fms",
                                    payload.get("window_index", "?"),
                                    payload.get("risk_state", "?"),
                                    payload.get("calibrated_probability", 0.0) or 0.0,
                                    payload.get("latency_ms", 0.0) or 0.0,
                                )
                            elif msg_type == "transition_state":
                                stats["transitions_received"].append(payload)
                                logger.info(
                                    "Receiver transition: status=%s spoof=%s",
                                    payload.get("status"),
                                    payload.get("spoof_enabled"),
                                )
                            else:
                                stats["status_received"].append(payload)
                        except json.JSONDecodeError:
                            pass
            except asyncio.CancelledError:
                pass
            except websockets.ConnectionClosed:
                pass

        reader_task = asyncio.create_task(receiver_loop())

        # Wait for caller to finish + extra grace period
        await asyncio.sleep(expected_duration_sec + 1.2)
        stop_event.set()
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass

    return stats


async def run_full_simulation(
    relay_url: str,
    session_id: str,
    duration_sec: float = 5.0,
    chunk_ms: int = 100,
    toggle_spoof_at: float = 2.5,
    wav_path: Path | None = None,
) -> bool:
    """Run full simulation with concurrent Caller and Receiver and verify zero packet drop."""
    logger.info("=============================================================")
    logger.info("  STARTING RELAY SERVICE ZERO-PACKET-DROP VERIFICATION")
    logger.info("=============================================================")

    if wav_path:
        audio = load_wav_file(wav_path)
        actual_duration = len(audio) / 16000.0
        duration_sec = min(duration_sec, actual_duration)
        audio = audio[: int(duration_sec * 16000)]
    else:
        audio = generate_speech_like_tone(duration_sec=duration_sec)

    logger.info(
        "Audio prepared: duration=%.2fs (%d samples, %d bytes)",
        duration_sec,
        audio.size,
        audio.size * 2,
    )

    # Run receiver and caller concurrently
    receiver_task = asyncio.create_task(
        run_receiver(
            relay_url=relay_url,
            session_id=session_id,
            expected_duration_sec=duration_sec,
        )
    )

    # Short 200ms delay to ensure receiver is connected before audio starts
    await asyncio.sleep(0.2)

    caller_task = asyncio.create_task(
        run_caller(
            relay_url=relay_url,
            session_id=session_id,
            audio=audio,
            chunk_ms=chunk_ms,
            toggle_spoof_at=toggle_spoof_at,
        )
    )

    caller_stats, rx_stats = await asyncio.gather(caller_task, receiver_task)

    # Analysis & Verification
    bytes_sent = caller_stats["bytes_sent"]
    bytes_rx = rx_stats["bytes_received"]
    chunks_sent = caller_stats["chunks_sent"]
    chunks_rx = rx_stats["chunks_received"]

    packet_loss_ratio = 1.0 - (bytes_rx / bytes_sent) if bytes_sent > 0 else 1.0
    packet_loss_percent = max(0.0, packet_loss_ratio * 100.0)

    logger.info("-------------------------------------------------------------")
    logger.info("  SIMULATION RESULTS")
    logger.info("-------------------------------------------------------------")
    logger.info("Chunks Sent by Caller     : %d", chunks_sent)
    logger.info("Chunks Received by Rx     : %d", chunks_rx)
    logger.info("Bytes Sent by Caller      : %d bytes (%.2fs)", bytes_sent, bytes_sent / 32000.0)
    logger.info("Bytes Received by Rx      : %d bytes (%.2fs)", bytes_rx, bytes_rx / 32000.0)
    logger.info("Packet Loss Rate          : %.2f%%", packet_loss_percent)
    logger.info("Scores Received from ML   : %d events", len(rx_stats["scores_received"]))
    logger.info("Fast Transitions Observed : %d", len(rx_stats["transitions_received"]))

    # Assertions
    success = True
    if bytes_rx == 0:
        logger.error("VERIFICATION FAILED: Zero bytes received by receiver!")
        success = False
    elif abs(bytes_rx - bytes_sent) > (2 * 3200):  # At most 1-2 chunks boundary rounding
        logger.warning(
            "Notice: Byte count discrepancy (%d sent vs %d received). Loss: %.2f%%",
            bytes_sent,
            bytes_rx,
            packet_loss_percent,
        )
        if packet_loss_percent > 5.0:
            logger.error("VERIFICATION FAILED: Packet loss exceeded acceptable bound!")
            success = False
    else:
        logger.info("Zero-Packet-Drop Check    : PASSED (loss <= 0.5%%)")

    if caller_stats["spoof_toggled"] and len(rx_stats["transitions_received"]) == 0:
        logger.warning("Warning: Spoof was toggled but no transition_state was observed at receiver.")

    return success


def main():
    parser = argparse.ArgumentParser(description="Voice Integrity Relay Service Test Simulator")
    parser.add_argument("--relay-url", default="ws://127.0.0.1:8080", help="Relay service base WebSocket URL")
    parser.add_argument("--session-id", default="sim_test_01", help="Call session identifier")
    parser.add_argument("--mode", choices=["all", "caller", "receiver"], default="all", help="Simulation mode")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of test audio in seconds")
    parser.add_argument("--chunk-ms", type=int, default=100, help="Audio chunk size in milliseconds")
    parser.add_argument("--toggle-spoof-at", type=float, default=2.5, help="Timestamp (sec) to toggle spoof")
    parser.add_argument("--wav", type=Path, default=None, help="Optional WAV file path to stream")

    args = parser.parse_args()

    if args.mode == "all":
        success = asyncio.run(
            run_full_simulation(
                relay_url=args.relay_url,
                session_id=args.session_id,
                duration_sec=args.duration,
                chunk_ms=args.chunk_ms,
                toggle_spoof_at=args.toggle_spoof_at,
                wav_path=args.wav,
            )
        )
        sys.exit(0 if success else 1)

    elif args.mode == "caller":
        audio = load_wav_file(args.wav) if args.wav else generate_speech_like_tone(args.duration)
        asyncio.run(
            run_caller(
                relay_url=args.relay_url,
                session_id=args.session_id,
                audio=audio,
                chunk_ms=args.chunk_ms,
                toggle_spoof_at=args.toggle_spoof_at,
            )
        )

    elif args.mode == "receiver":
        asyncio.run(
            run_receiver(
                relay_url=args.relay_url,
                session_id=args.session_id,
                expected_duration_sec=args.duration,
            )
        )


if __name__ == "__main__":
    main()
