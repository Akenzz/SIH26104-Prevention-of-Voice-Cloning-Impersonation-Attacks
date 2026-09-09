"""Unit tests for CallSession and SessionManager."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from config import RelaySettings
from session import CallSession, SessionManager


@pytest.mark.asyncio
async def test_session_lifecycle():
    settings = RelaySettings(backend_ws_url="ws://127.0.0.1:9999/ws")
    session = CallSession(session_id="test_01", settings=settings)

    # Mock backend client methods so no real network is needed for this unit test
    session.backend_client.start = AsyncMock()
    session.backend_client.stop = AsyncMock()
    session.backend_client.send_audio = AsyncMock(return_value=True)
    session.backend_client.flush_on_spoof_toggle = AsyncMock()

    await session.start()
    assert session.is_active is True

    # Register mock caller and receiver
    mock_caller = MagicMock()
    mock_caller.send_json = AsyncMock()

    mock_receiver = MagicMock()
    mock_receiver.send_bytes = AsyncMock()
    mock_receiver.send_json = AsyncMock()

    await session.register_caller(mock_caller)
    await session.register_receiver(mock_receiver)
    assert session.caller_ws == mock_caller
    assert mock_receiver in session.receivers

    # Enqueue a PCM chunk (e.g. 1600 samples = 3200 bytes)
    dummy_chunk = b"\x00\x00" * 1600
    await session.enqueue_audio(dummy_chunk)

    # Let the worker process the chunk
    await asyncio.sleep(0.1)

    # Receiver should have received audio bytes
    assert mock_receiver.send_bytes.called
    assert session.backend_client.send_audio.called

    # Toggle spoof -> must call flush_on_spoof_toggle and broadcast transition_state
    await session.set_spoof(True)
    assert session.spoof_enabled is True
    assert session.backend_client.flush_on_spoof_toggle.called

    # Verify receiver got transition_state message
    rx_json_calls = [call.args[0] for call in mock_receiver.send_json.call_args_list]
    transition_msgs = [m for m in rx_json_calls if m.get("type") == "transition_state"]
    assert len(transition_msgs) >= 1
    assert transition_msgs[0]["status"] == "switching"
    assert transition_msgs[0]["spoof_enabled"] is True

    await session.stop()
    assert session.is_active is False


@pytest.mark.asyncio
async def test_session_manager():
    settings = RelaySettings(backend_ws_url="ws://127.0.0.1:9999/ws")
    mgr = SessionManager(settings)

    s1 = await mgr.get_or_create_session("call_a")
    assert s1.session_id == "call_a"
    assert "call_a" in mgr.sessions

    s2 = await mgr.get_or_create_session("call_a")
    assert s1 is s2

    await mgr.close_session("call_a")
    assert "call_a" not in mgr.sessions
    await mgr.close_all()
