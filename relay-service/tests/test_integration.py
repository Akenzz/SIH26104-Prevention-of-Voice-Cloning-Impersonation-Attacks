"""Full end-to-end integration test for Relay Service: Caller -> Relay -> Receiver + Backend."""

import asyncio
import json
import pytest
import uvicorn
import websockets

from config import RelaySettings
from server import app, session_manager


@pytest.mark.asyncio
async def test_full_relay_flow():
    # 1. Start mock detection backend
    mock_backend_received = []

    async def mock_backend_handler(ws):
        # Handshake
        msg = await ws.recv()
        data = json.loads(msg)
        assert data.get("type") == "start"
        await ws.send(json.dumps({"type": "ready"}))

        while True:
            try:
                frame = await ws.recv()
                if isinstance(frame, bytes):
                    mock_backend_received.append(len(frame))
                    # Emit detection score
                    await ws.send(json.dumps({
                        "type": "score",
                        "event": "window_scored",
                        "sequence_number": len(mock_backend_received),
                        "window_index": len(mock_backend_received),
                        "risk_state": "spoof" if len(mock_backend_received) > 2 else "bonafide",
                        "calibrated_probability": 0.88 if len(mock_backend_received) > 2 else 0.05,
                        "weighted_probability": 0.88 if len(mock_backend_received) > 2 else 0.05,
                        "per_expert_probability": {"hybrid": 0.89, "ssl": 0.85},
                        "scores": {},
                        "latency_ms": 12.0,
                    }))
                elif isinstance(frame, str):
                    p = json.loads(frame)
                    if p.get("type") == "start":
                        await ws.send(json.dumps({"type": "ready"}))
            except websockets.ConnectionClosed:
                break

    backend_server = await websockets.serve(mock_backend_handler, "127.0.0.1", 0)
    backend_port = backend_server.sockets[0].getsockname()[1]
    backend_url = f"ws://127.0.0.1:{backend_port}"

    # 2. Configure relay app with the mock backend URL
    relay_port = 8899
    custom_settings = RelaySettings(
        host="127.0.0.1",
        port=relay_port,
        backend_ws_url=backend_url,
        sample_rate=16000,
        encoding="pcm_s16le",
        chunk_ms=100,
    )
    # Update global settings and session manager
    session_manager.settings = custom_settings

    # Run uvicorn server in background
    config = uvicorn.Config(app, host="127.0.0.1", port=relay_port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Wait for server to start
    await asyncio.sleep(0.5)

    try:
        session_id = "test_integ_room"
        caller_url = f"ws://127.0.0.1:{relay_port}/ws/caller?session_id={session_id}"
        receiver_url = f"ws://127.0.0.1:{relay_port}/ws/receiver?session_id={session_id}"

        receiver_frames = []
        receiver_scores = []
        receiver_transitions = []
        stop_rx = asyncio.Event()

        # 3. Connect receiver
        async def rx_worker():
            async with websockets.connect(receiver_url) as rx_ws:
                while not stop_rx.is_set():
                    try:
                        msg = await asyncio.wait_for(rx_ws.recv(), timeout=0.1)
                        if isinstance(msg, bytes):
                            receiver_frames.append(msg)
                        else:
                            data = json.loads(msg)
                            if data.get("type") == "score" or data.get("event") == "window_scored":
                                receiver_scores.append(data)
                            elif data.get("type") == "transition_state":
                                receiver_transitions.append(data)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.ConnectionClosed:
                        break

        rx_task = asyncio.create_task(rx_worker())
        await asyncio.sleep(0.2)

        # 4. Connect caller and stream audio
        async with websockets.connect(caller_url) as caller_ws:
            ready_msg = await caller_ws.recv()
            assert "ready" in ready_msg

            # Send 3 bonafide chunks (1600 samples = 3200 bytes each)
            chunk = b"\x10\x00" * 1600
            for _ in range(3):
                await caller_ws.send(chunk)
                await asyncio.sleep(0.05)

            # Toggle spoof ON
            await caller_ws.send(json.dumps({"type": "toggle_spoof", "enabled": True}))
            await asyncio.sleep(0.05)

            # Send 3 spoofed chunks
            for _ in range(3):
                await caller_ws.send(chunk)
                await asyncio.sleep(0.05)

            await asyncio.sleep(0.5)

        stop_rx.set()
        await rx_task

        # Verify results
        assert len(receiver_frames) >= 4, f"Expected >= 4 frames received, got {len(receiver_frames)}"
        assert len(receiver_transitions) >= 1, "Expected at least 1 transition_state event"
        assert receiver_transitions[0]["status"] == "switching"
        assert receiver_transitions[0]["spoof_enabled"] is True

        assert len(receiver_scores) >= 1, f"Expected scores from backend, got {len(receiver_scores)}"
        first_score = receiver_scores[0]
        assert "risk_state" in first_score
        assert "calibrated_probability" in first_score
        assert "weighted_probability" in first_score

    finally:
        server.should_exit = True
        await server_task
        backend_server.close()
        await backend_server.wait_closed()
        await session_manager.close_all()
