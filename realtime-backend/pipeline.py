"""Per-connection scoring pipeline: decode → window → experts → fuse → calibrate → policy."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from audio.quality import QualityResult, assess_window, decode_pcm
from audio.resample import to_mono, to_target_rate
from audio.ring_buffer import RingBuffer
from calibration import Calibrator
from config import Settings
from experts.protocol import Expert, Score
from fusion import FusionConfig, fuse_logits
from messages import build_message
from policy import PolicyConfig, decide
from smoothing import ExponentialMovingAverage

logger = logging.getLogger("realtime_backend.pipeline")

VALID_ENCODINGS = {"pcm_s16le", "s16le", "int16", "pcm_f32le", "f32le", "float32"}


@dataclass
class SessionConfig:
    sample_rate: int
    encoding: str
    channels: int = 1
    next_client_seq: int | None = None
    binary_seq: bool = False


def process_single_window(
    window: np.ndarray,
    settings: Settings,
    experts: dict[str, Expert],
    fusion: FusionConfig,
    calibrator: Calibrator,
    policy: PolicyConfig,
    windows_scored: int,
    smoothed_probability: float | None = None, # If None, will use raw prob
    dropped_frames: bool = False,
) -> tuple[str, str, str | None, dict[str, Score], float | None, float | None, QualityResult]:
    """Core logic to assess, score, fuse, calibrate, and decide on a single window."""
    quality = assess_window(
        window,
        silence_rms=settings.silence_rms,
        clip_abs=settings.clip_abs,
        clip_fraction=settings.clip_fraction,
    )
    if dropped_frames:
        quality = QualityResult(False, "dropped_or_reordered")

    if not quality.ok:
        state, action, flag = decide(
            quality_ok=False,
            quality_reason=quality.reason,
            windows_scored=windows_scored,
            smoothed_probability=smoothed_probability,
            config=policy,
        )
        return state, action, flag, {}, None, None, quality

    scores: dict[str, Score] = {}
    for name, expert in experts.items():
        scores[name] = expert.score(window)

    fused, _used = fuse_logits(
        scores,
        fusion,
        mode=settings.fusion_mode,
        single_expert=settings.single_expert,
    )
    probability = calibrator.probability(fused)
    
    # For single-shot (no smoothing), use the raw probability
    if smoothed_probability is None:
        smoothed_probability = probability

    state, action, flag = decide(
        quality_ok=True,
        quality_reason=None,
        windows_scored=windows_scored,
        smoothed_probability=smoothed_probability,
        config=policy,
    )
    return state, action, flag, scores, fused, probability, quality

@dataclass
class ConnectionState:
    settings: Settings
    experts: dict[str, Expert]
    fusion: FusionConfig
    calibrator: Calibrator
    policy: PolicyConfig
    session: SessionConfig | None = None
    buffer: RingBuffer | None = None
    ema: ExponentialMovingAverage = field(init=False)
    out_seq: int = 0
    windows_scored: int = 0
    resample_warned: bool = False
    started: bool = False

    def __post_init__(self) -> None:
        self.ema = ExponentialMovingAverage(self.settings.ema_alpha)

    def apply_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        sample_rate = int(payload.get("sample_rate") or payload.get("sampleRate") or 0)
        encoding = str(payload.get("encoding") or payload.get("format") or "pcm_s16le").lower()
        channels = int(payload.get("channels") or 1)
        binary_seq = bool(payload.get("binary_seq") or payload.get("frame_seq") or False)
        if sample_rate <= 0:
            raise ValueError("start message must declare a positive sample_rate")
        if encoding not in VALID_ENCODINGS:
            raise ValueError(f"unsupported encoding {encoding!r}")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        self.session = SessionConfig(
            sample_rate=sample_rate,
            encoding=encoding,
            channels=channels,
            binary_seq=binary_seq,
        )
        self.buffer = RingBuffer(self.settings.window_samples, self.settings.hop_samples)
        self.ema.reset()
        self.out_seq = 0
        self.windows_scored = 0
        self.resample_warned = False
        self.started = True
        if sample_rate != self.settings.target_sample_rate:
            logger.info(
                "Client declared %d Hz; will resample to %d Hz (confirmed once on first frame)",
                sample_rate,
                self.settings.target_sample_rate,
            )
        return {
            "type": "ready",
            "accepted_sample_rate": sample_rate,
            "target_sample_rate": self.settings.target_sample_rate,
            "window_sec": self.settings.window_sec,
            "hop_sec": self.settings.hop_sec,
            "encoding": encoding,
            "channels": channels,
            "experts": list(self.experts),
            "fusion_mode": self.settings.fusion_mode,
        }

    def ingest_binary(self, payload: bytes) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        if not self.started or self.session is None or self.buffer is None:
            return [self._error("send a JSON start message before audio frames")]

        frame = payload
        dropped = False
        if self.session.binary_seq:
            if len(payload) < 4:
                q = QualityResult(False, "decode_failure")
                return [self._unavailable(q, t0, scores={}, fused=None, window_index=self.windows_scored)]
            client_seq = int.from_bytes(payload[:4], "little", signed=False)
            frame = payload[4:]
            dropped = self._consume_client_seq(client_seq)
            if dropped:
                q = QualityResult(False, "dropped_or_reordered")
                return [self._unavailable(q, t0, scores={}, fused=None, window_index=self.windows_scored)]

        samples, decode_q = decode_pcm(frame, self.session.encoding, self.session.channels)
        if samples is None:
            return [self._unavailable(decode_q, t0, scores={}, fused=None, window_index=self.windows_scored)]
        try:
            mono = to_mono(samples, self.session.channels)
            resampled, did_resample = to_target_rate(
                mono, self.session.sample_rate, self.settings.target_sample_rate
            )
        except ValueError:
            q = QualityResult(False, "decode_failure")
            return [self._unavailable(q, t0, scores={}, fused=None, window_index=self.windows_scored)]

        if did_resample and not self.resample_warned:
            self.resample_warned = True
            logger.warning(
                "Resampling active: %d Hz -> %d Hz (logged once per connection)",
                self.session.sample_rate,
                self.settings.target_sample_rate,
            )

        windows = self.buffer.push(resampled)
        if not windows:
            return []

        messages: list[dict[str, Any]] = []
        for window in windows:
            messages.append(self._score_window(window, t0, dropped_frames=dropped))
            dropped = False
            t0 = time.perf_counter()
        return messages

    def ingest_json_frame(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Optional JSON audio frame with an explicit client sequence number."""
        t0 = time.perf_counter()
        if not self.started or self.session is None or self.buffer is None:
            return [self._error("send a JSON start message before audio frames")]

        client_seq = payload.get("sequence_number", payload.get("seq"))
        dropped = self._consume_client_seq(client_seq if client_seq is None else int(client_seq))
        if dropped:
            q = QualityResult(False, "dropped_or_reordered")
            return [self._unavailable(q, t0, scores={}, fused=None, window_index=self.windows_scored)]

        raw = payload.get("pcm") or payload.get("audio")
        if raw is None:
            q = QualityResult(False, "decode_failure")
            return [self._unavailable(q, t0, scores={}, fused=None, window_index=self.windows_scored)]

        encoding = str(payload.get("encoding") or self.session.encoding)
        if isinstance(raw, str):
            import base64

            try:
                blob = base64.b64decode(raw)
            except (ValueError, TypeError):
                q = QualityResult(False, "decode_failure")
                return [self._unavailable(q, t0, scores={}, fused=None, window_index=self.windows_scored)]
            samples, decode_q = decode_pcm(blob, encoding, self.session.channels)
        elif isinstance(raw, list):
            samples = np.asarray(raw, dtype=np.float32)
            decode_q = QualityResult(True, None)
        else:
            samples, decode_q = None, QualityResult(False, "decode_failure")

        if samples is None or not decode_q.ok:
            return [self._unavailable(decode_q, t0, scores={}, fused=None, window_index=self.windows_scored)]

        try:
            mono = to_mono(samples, int(payload.get("channels") or self.session.channels))
            source_rate = int(payload.get("sample_rate") or self.session.sample_rate)
            resampled, did_resample = to_target_rate(
                mono, source_rate, self.settings.target_sample_rate
            )
            if did_resample and not self.resample_warned:
                self.resample_warned = True
                logger.warning(
                    "Resampling active: %d Hz -> %d Hz (logged once per connection)",
                    source_rate,
                    self.settings.target_sample_rate,
                )
        except ValueError:
            q = QualityResult(False, "decode_failure")
            return [self._unavailable(q, t0, scores={}, fused=None, window_index=self.windows_scored)]

        windows = self.buffer.push(resampled)
        return [self._score_window(w, t0, dropped_frames=False) for w in windows]

    def _consume_client_seq(self, client_seq: int | None) -> bool:
        if self.session is None or client_seq is None:
            return False
        if self.session.next_client_seq is None:
            self.session.next_client_seq = client_seq + 1
            return False
        if client_seq != self.session.next_client_seq:
            logger.warning(
                "Client sequence gap: expected %s got %s",
                self.session.next_client_seq,
                client_seq,
            )
            self.session.next_client_seq = client_seq + 1
            return True
        self.session.next_client_seq = client_seq + 1
        return False


    def _score_window(
        self,
        window: np.ndarray,
        t0: float,
        *,
        dropped_frames: bool,
    ) -> dict[str, Any]:
        
        # First we need the raw probability to update EMA. 
        # But we need fusion to get the probability.
        # We can run the core sequence, but we want to pass the updated EMA back to `decide`.
        # Actually, `process_single_window` handles `decide`. So we run it once, 
        # get the raw probability, update EMA, and if quality is ok, we'll want `decide` 
        # with the EMA.
        
        # We can just do quality assessment here, or extract the fusion math.
        # It's cleaner to let `process_single_window` do everything, but skip its `decide` 
        # and do it here if we want EMA.
        # Or modify `process_single_window` to accept a callback for smoothing?
        # Let's simplify: `process_single_window` returns all intermediate values.
        
        state, action, flag, scores, fused, probability, quality = process_single_window(
            window=window,
            settings=self.settings,
            experts=self.experts,
            fusion=self.fusion,
            calibrator=self.calibrator,
            policy=self.policy,
            windows_scored=self.windows_scored + 1,
            smoothed_probability=None, # pass None first, we'll re-decide if ok
            dropped_frames=dropped_frames
        )
        
        if not quality.ok:
            return self._unavailable(
                quality, t0, scores={}, fused=None, window_index=self.windows_scored
            )

        smoothed = self.ema.update(probability)
        self.windows_scored += 1
        
        state, action, flag = decide(
            quality_ok=True,
            quality_reason=None,
            windows_scored=self.windows_scored,
            smoothed_probability=smoothed,
            config=self.policy,
        )
        
        return self._emit(
            state=state,
            action=action,
            smoothed=smoothed,
            fused=fused,
            scores=scores,
            dropped=False,
            audio_quality=flag,
            window_index=self.windows_scored,
            t0=t0,
        )

    def _unavailable(
        self,
        quality: QualityResult,
        t0: float,
        *,
        scores: dict[str, Score],
        fused: float | None,
        window_index: int,
    ) -> dict[str, Any]:
        state, action, flag = decide(
            quality_ok=False,
            quality_reason=quality.reason,
            windows_scored=self.windows_scored,
            smoothed_probability=self.ema.value,
            config=self.policy,
        )
        return self._emit(
            state=state,
            action=action,
            smoothed=None,
            fused=fused,
            scores=scores,
            dropped=quality.reason == "dropped_or_reordered",
            audio_quality=flag,
            window_index=window_index,
            t0=t0,
        )

    def _emit(
        self,
        *,
        state: str,
        action: str,
        smoothed: float | None,
        fused: float | None,
        scores: dict[str, Score],
        dropped: bool,
        audio_quality: str | None,
        window_index: int,
        t0: float,
    ) -> dict[str, Any]:
        self.out_seq += 1
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return build_message(
            sequence_number=self.out_seq,
            risk_state=state,
            recommended_action=action,
            smoothed_probability=smoothed,
            fused_logit=fused,
            scores=scores,
            threshold_version=self.policy.version,
            calibrator_version=self.calibrator.version,
            fusion_version=self.fusion.version,
            fusion_mode=self.settings.fusion_mode,
            dropped_frames=dropped,
            audio_quality=audio_quality,
            window_index=window_index,
            latency_ms=latency_ms,
        )

    def _error(self, detail: str) -> dict[str, Any]:
        self.out_seq += 1
        return {
            "type": "error",
            "sequence_number": self.out_seq,
            "risk_state": "unavailable",
            "recommended_action": self.policy.actions.get(
                "unavailable",
                "Audio quality or stream integrity failed. Do not treat this as a real-speech score.",
            ),
            "smoothed_probability": None,
            "fused_logit": None,
            "raw_per_expert_scores": {},
            "model_version": {},
            "threshold_version": self.policy.version,
            "calibrator_version": self.calibrator.version,
            "fusion_version": self.fusion.version,
            "fusion_mode": self.settings.fusion_mode,
            "dropped_frames": False,
            "audio_quality": "protocol_error",
            "detail": detail,
            "window_index": self.windows_scored,
            "latency_ms": None,
        }
