"""Unit tests for RealtimeBackendClient protocol handshake and score parsing."""

import asyncio
import json
import pytest
import websockets

from backend_client import RealtimeBackendClient


@pytest.mark.asyncio
async def test_backend_handshake_and_scoring():
    # 1. Spawn a lightweight mock backend server
    received_messages = []
    ready_sent = asyncio.Event()

    async def mock_backend_handler(ws):
        # 1. Expect start message
        msg = await ws.recv()
        data = json.loads(msg)
        received_messages.append(data)
        assert data.get("type") == "start"
        assert data.get("sample_rate") == 16000
        assert data.get("encoding") == "pcm_s16le"

        # Reply ready
        await ws.send(json.dumps({
            "type": "ready",
            "accepted_sample_rate": 16000,
            "target_sample_rate": 16000,
        }))
        ready_sent.set()

        # Listen for audio or flush
        while True:
            try:
                frame = await ws.recv()
                if isinstance(frame, bytes):
                    received_messages.append({"bytes_len": len(frame)})
                    # Emit a mock score
                    await ws.send(json.dumps({
                        "type": "score",
                        "sequence_number": 1,
                        "risk_state": "bonafide",
                        "smoothed_probability": 0.05,
                        "scores": {
                            "wavlm": {"probability": 0.04, "logit": -3.0},
                            "hybrid": {"probability": 0.06, "logit": -2.8},
                        },
                        "latency_ms": 15.2,
                    }))
                elif isinstance(frame, str):
                    payload = json.loads(frame)
                    received_messages.append(payload)
                    if payload.get("type") == "start":
                        # Flush acknowledgment
                        await ws.send(json.dumps({"type": "ready"}))
            except websockets.ConnectionClosed:
                break

    # Start mock server on an ephemeral port
    server = await websockets.serve(mock_backend_handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    backend_url = f"ws://127.0.0.1:{port}"

    scores_received = []
    score_event = asyncio.Event()

    async def on_score(score_data):
        scores_received.append(score_data)
        score_event.set()

    client = RealtimeBackendClient(
        backend_url=backend_url,
        sample_rate=16000,
        encoding="pcm_s16le",
    )
    client.add_score_callback(on_score)

    try:
        await client.start()
        await ready_sent.wait()
        assert client.is_ready is True

        # Send binary audio frame
        dummy_audio = b"\x00\x00" * 800
        sent = await client.send_audio(dummy_audio)
        assert sent is True

        # Wait for score callback
        await asyncio.wait_for(score_event.wait(), timeout=3.0)
        assert len(scores_received) == 1

        score = scores_received[0]
        # Verify required normalized fields
        assert score["event"] == "window_scored"
        assert score["type"] == "score"
        assert score["risk_state"] == "bonafide"
        assert score["calibrated_probability"] == 0.05
        assert score["weighted_probability"] == 0.05
        assert "wavlm" in score["per_expert_probability"]
        assert score["per_expert_probability"]["wavlm"] == 0.04

        # Test fast transition flush on spoof toggle
        await client.flush_on_spoof_toggle()
        await asyncio.sleep(0.1)
        # Check that mock backend received second start message
        starts = [m for m in received_messages if isinstance(m, dict) and m.get("type") == "start"]
        assert len(starts) == 2

    finally:
        await client.stop()
        server.close()
        await server.wait_closed()
