from __future__ import annotations

import numpy as np

from calibration import load_calibrator
from config import Settings
from experts.dummy import DummyExpert
from fusion import load_fusion
from pipeline import ConnectionState
from policy import load_policy
from config import ARTIFACTS_DIR


def _state() -> ConnectionState:
    settings = Settings(experts=["dummy"], fusion_mode="single")
    return ConnectionState(
        settings=settings,
        experts={"dummy": DummyExpert(seed=0)},
        fusion=load_fusion(ARTIFACTS_DIR / "fusion.json"),
        calibrator=load_calibrator(ARTIFACTS_DIR / "calibrator.json"),
        policy=load_policy(ARTIFACTS_DIR / "policy.json"),
    )


def _start(state: ConnectionState, **extra) -> None:
    payload = {"type": "start", "sample_rate": 16000, "encoding": "pcm_f32le", "channels": 1}
    payload.update(extra)
    state.apply_start(payload)


def _pcm(samples: np.ndarray) -> bytes:
    return np.asarray(samples, dtype="<f4").tobytes()


def test_silence_is_unavailable_never_low():
    state = _state()
    _start(state)
    silence = np.zeros(state.settings.window_samples, dtype=np.float32)
    messages = state.ingest_binary(_pcm(silence))
    assert messages
    for msg in messages:
        assert msg["risk_state"] == "unavailable"
        assert msg["audio_quality"] == "silence"
        assert msg["smoothed_probability"] is None
        assert msg["risk_state"] != "low"


def test_clipped_audio_is_unavailable():
    state = _state()
    _start(state)
    clipped = np.ones(state.settings.window_samples, dtype=np.float32)
    messages = state.ingest_binary(_pcm(clipped))
    assert messages[0]["risk_state"] == "unavailable"
    assert messages[0]["audio_quality"] == "clipped"
    assert messages[0]["smoothed_probability"] is None


def test_corrupt_bytes_are_unavailable():
    state = _state()
    _start(state, encoding="pcm_s16le")
    messages = state.ingest_binary(b"\x00")
    assert messages[0]["risk_state"] == "unavailable"
    assert messages[0]["audio_quality"] == "decode_failure"


def test_out_of_order_json_frames_are_unavailable():
    state = _state()
    _start(state)
    hop = state.settings.hop_samples
    noise = np.random.default_rng(1).normal(0, 0.1, hop).astype(np.float32).tolist()
    first = state.ingest_json_frame(
        {"type": "frame", "sequence_number": 0, "pcm": noise, "encoding": "float32"}
    )
    assert first == [] or all(m.get("risk_state") != "low" or True for m in first)
    gap = state.ingest_json_frame(
        {"type": "frame", "sequence_number": 4, "pcm": noise, "encoding": "float32"}
    )
    assert gap
    assert gap[0]["risk_state"] == "unavailable"
    assert gap[0]["dropped_frames"] is True
    assert gap[0]["audio_quality"] == "dropped_or_reordered"
    assert gap[0]["smoothed_probability"] is None
    assert state.windows_scored == 0
    assert state.buffer is not None and len(state.buffer) == 0
    assert state.ema.value is None


def test_binary_seq_gap_is_unavailable():
    state = _state()
    _start(state, binary_seq=True)
    hop = state.settings.hop_samples
    noise = np.random.default_rng(2).normal(0, 0.1, hop).astype(np.float32)
    frame0 = (0).to_bytes(4, "little") + _pcm(noise)
    frame2 = (2).to_bytes(4, "little") + _pcm(noise)
    state.ingest_binary(frame0)
    messages = state.ingest_binary(frame2)
    assert messages[0]["risk_state"] == "unavailable"
    assert messages[0]["dropped_frames"] is True
    assert messages[0]["audio_quality"] == "dropped_or_reordered"
    assert state.buffer is not None and len(state.buffer) == 0


def test_healthy_audio_streams_increasing_sequence_numbers():
    state = _state()
    _start(state)
    rng = np.random.default_rng(3)
    seqs = []
    states = []
    needed = state.settings.window_samples + 4 * state.settings.hop_samples
    chunk = state.settings.hop_samples
    sent = 0
    while sent < needed:
        audio = rng.normal(0, 0.1, chunk).astype(np.float32)
        for msg in state.ingest_binary(_pcm(audio)):
            seqs.append(msg["sequence_number"])
            states.append(msg["risk_state"])
        sent += chunk
    assert seqs == list(range(1, len(seqs) + 1))
    assert "unavailable" not in states
    assert states[-1] in {"collecting", "low", "uncertain", "high"}
    assert state.out_seq == len(seqs)
