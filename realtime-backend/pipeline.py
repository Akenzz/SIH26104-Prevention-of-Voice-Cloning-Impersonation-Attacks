"""Per-connection scoring pipeline: decode → window → experts → fuse → calibrate → policy."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from audio.quality import QualityResult, assess_window, decode_pcm
from audio.resample import StreamingPolyphaseResampler, to_mono
from audio.ring_buffer import RingBuffer
from calibration import Calibrator
from config import Settings
from experts.protocol import Expert, Score
from fusion import FusionConfig, decision_expert, fuse_logits
from messages import build_message
from policy import PolicyConfig, decide
from smoothing import ExponentialMovingAverage

import joblib
from config import ARTIFACTS_DIR

logger = logging.getLogger("realtime_backend.pipeline")

_LR_MODEL = None
try:
    _LR_MODEL = joblib.load(ARTIFACTS_DIR / "fusion_lr.joblib")
except Exception as e:
    # Missing/unreadable LR model must NOT crash import: the pipeline still runs
    # every expert and shows each side card; only the lr_fusion headline degrades
    # (it falls back to the calibrated single-expert logit).
    logger.warning(f"Could not load LR model: {e}")

VALID_ENCODINGS = {"pcm_s16le", "s16le", "int16", "pcm_f32le", "f32le", "float32"}


@dataclass
class SessionConfig:
    sample_rate: int
    encoding: str
    channels: int = 1
    next_client_seq: int | None = None
    binary_seq: bool = False


@dataclass
class WindowResult:
    """Everything one scored window produced.

    `expert_probabilities` is per-expert and each entry uses that expert's OWN
    calibrator; `probability` is the decision probability from the fused/single
    logit and is the only one the risk band is derived from.
    """

    risk_state: str
    recommended_action: str
    audio_quality: str | None
    scores: dict[str, Score]
    fused_logit: float | None
    probability: float | None
    quality: QualityResult
    expert_probabilities: dict[str, float] = field(default_factory=dict)
    lr_probability: float | None = None


def process_single_window(
    window: np.ndarray,
    settings: Settings,
    experts: dict[str, Expert],
    fusion: FusionConfig,
    calibrator: Calibrator,
    policy: PolicyConfig,
    windows_scored: int,
    smoothed_probability: float | None = None,  # If None, will use raw prob
    dropped_frames: bool = False,
    expert_calibrators: dict[str, Calibrator] | None = None,
) -> WindowResult:
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
        return WindowResult(state, action, flag, {}, None, None, quality)

    scores: dict[str, Score] = {}
    for name, expert in experts.items():
        scores[name] = expert.score(window)

    fused, _used = fuse_logits(
        scores,
        fusion,
        mode=settings.fusion_mode,
        single_expert=settings.single_expert,
    )
    # `fused` is ONE expert's raw logit in `single` mode -- and in `lr_fusion`
    # too, where it is the value used whenever the LR model is unavailable. A raw
    # logit has to be read on its OWN expert's Platt scale. The global
    # calibrator belongs to whatever CALIBRATOR_PATH points at, which in the
    # committed default (lr_fusion, CALIBRATOR_PATH=calibrator_hybrid_newclips)
    # is a DIFFERENT expert than `hybrid`: reading hybrid's logits on hybrid_nc's
    # scale moves the low/uncertain boundary from logit 1.66 to 0.59 and
    # uncertain/high from 4.42 to 3.00, i.e. a different verdict over ~15% of the
    # logit range, all of it in range and plausible-looking.
    # calibration.assert_single_expert_calibrator() already forces the two to
    # agree in `single` mode, so this is a no-op there; it is what keeps the
    # lr_fusion FALLBACK honest. Genuine combinations (`fused`, `heuristic*`)
    # return None and keep the global calibrator.
    picked = decision_expert(
        scores, mode=settings.fusion_mode, single_expert=settings.single_expert
    )
    decision_calibrator = calibrator
    if picked is not None:
        decision_calibrator = (expert_calibrators or {}).get(picked, calibrator)
    probability = decision_calibrator.probability(fused)

    # Each expert gets its own calibrator; falling back to the global one would
    # read a foreign logit scale, so we only report what we can calibrate.
    expert_probabilities: dict[str, float] = {}
    for name, score in scores.items():
        cal = (expert_calibrators or {}).get(name, calibrator)
        expert_probabilities[name] = float(cal.probability(float(score["logit"])))

    # Calculate LR probability for frontend visualization if available.
    # When mode == lr_fusion, this also becomes the primary decision probability.
    lr_probability = None
    if _LR_MODEL is not None and "wavlm" in scores and "hybrid" in scores and "ssl" in scores:
        w_log = float(scores["wavlm"]["logit"])
        h_log = float(scores["hybrid"]["logit"])
        s_log = float(scores["ssl"]["logit"])
        try:
            lr_probability = float(_LR_MODEL.predict_proba(np.array([[w_log, h_log, s_log]]))[0, 1])
        except Exception:
            pass

    if settings.fusion_mode == "lr_fusion":
        if lr_probability is not None:
            probability = lr_probability
        # else fall back to the fused logit already computed
    elif settings.fusion_mode == "heuristic_avg":
        p_l = expert_probabilities.get("hybrid", 0.0)
        p_s = expert_probabilities.get("ssl", 0.0)
        probability = float((p_l + p_s) / 2.0)
        fused = None

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
    return WindowResult(
        state, action, flag, scores, fused, probability, quality, expert_probabilities, lr_probability
    )


@dataclass
class ConnectionState:
    settings: Settings
    experts: dict[str, Expert]
    fusion: FusionConfig
    calibrator: Calibrator
    policy: PolicyConfig
    expert_calibrators: dict[str, Calibrator] = field(default_factory=dict)
    session: SessionConfig | None = None
    buffer: RingBuffer | None = None
    # One resampler per connection: resampling each incoming frame independently
    # restarts the anti-aliasing filter and re-rounds the output length, which
    # drifts the stream and corrupts the audio before any expert sees it.
    resampler: StreamingPolyphaseResampler | None = None
    ema: ExponentialMovingAverage = field(init=False)
    lr_ema: ExponentialMovingAverage = field(init=False)
    expert_emas: dict[str, ExponentialMovingAverage] = field(init=False)
    out_seq: int = 0
    windows_scored: int = 0
    resample_warned: bool = False
    started: bool = False

    def __post_init__(self) -> None:
        self.ema = ExponentialMovingAverage(self.settings.ema_alpha)
        self.lr_ema = ExponentialMovingAverage(self.settings.ema_alpha)
        self.expert_emas = {
            name: ExponentialMovingAverage(self.settings.ema_alpha) for name in self.experts
        }

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
        self.resampler = StreamingPolyphaseResampler(sample_rate, self.settings.target_sample_rate)
        self.ema.reset()
        self.lr_ema.reset()
        for e in self.expert_emas.values():
            e.reset()
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
                self._reset_after_stream_discontinuity()
                q = QualityResult(False, "dropped_or_reordered")
                return [self._unavailable(q, t0, scores={}, fused=None, window_index=self.windows_scored)]

        samples, decode_q = decode_pcm(frame, self.session.encoding, self.session.channels)
        if samples is None:
            return [self._unavailable(decode_q, t0, scores={}, fused=None, window_index=self.windows_scored)]
        try:
            mono = to_mono(samples, self.session.channels)
            if self.resampler is None:
                raise RuntimeError("streaming resampler was not initialized")
            resampled = self.resampler.process(mono)
            did_resample = self.resampler.did_resample
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
            self._reset_after_stream_discontinuity()
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
            # The resampler carries filter state for one fixed rate pair, so the
            # rate cannot change mid-stream. Ask the client to restart instead of
            # silently splicing two sample rates into one window.
            if source_rate != self.session.sample_rate:
                logger.warning(
                    "Frame declared %d Hz but the stream started at %d Hz; rejecting frame",
                    source_rate,
                    self.session.sample_rate,
                )
                raise ValueError(
                    "sample_rate cannot change within a stream; send a new start message"
                )
            if self.resampler is None:
                raise RuntimeError("streaming resampler was not initialized")
            resampled = self.resampler.process(mono)
            did_resample = self.resampler.did_resample
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

    def _reset_after_stream_discontinuity(self) -> None:
        """Stop a window from joining audio from opposite sides of a gap.

        After dropped or reordered frames the buffered tail, the resampler's
        filter state, and the smoothed score all describe audio that is no
        longer contiguous with what is arriving now.
        """
        if self.buffer is not None:
            self.buffer.reset()
        if self.resampler is not None:
            self.resampler.reset()
        self.ema.reset()
        self.lr_ema.reset()
        for e in self.expert_emas.values():
            e.reset()
        self.windows_scored = 0

    def _score_window(
        self,
        window: np.ndarray,
        t0: float,
        *,
        dropped_frames: bool,
    ) -> dict[str, Any]:
        # process_single_window decides with the RAW probability; we re-decide
        # below with the EMA-smoothed one, which is what the contract reports.
        result = process_single_window(
            window=window,
            settings=self.settings,
            experts=self.experts,
            fusion=self.fusion,
            calibrator=self.calibrator,
            policy=self.policy,
            windows_scored=self.windows_scored + 1,
            smoothed_probability=None,
            dropped_frames=dropped_frames,
            expert_calibrators=self.expert_calibrators,
        )

        if not result.quality.ok:
            return self._unavailable(
                result.quality, t0, scores={}, fused=None, window_index=self.windows_scored
            )

        smoothed = self.ema.update(result.probability)
        lr_smoothed = self.lr_ema.update(result.lr_probability) if result.lr_probability is not None else None
        self.windows_scored += 1

        # Smooth each expert on its own track so the per-model cards are as
        # stable as the headline number instead of jittering window to window.
        expert_smoothed: dict[str, float] = {}
        for name, prob in result.expert_probabilities.items():
            ema = self.expert_emas.setdefault(name, ExponentialMovingAverage(self.settings.ema_alpha))
            expert_smoothed[name] = float(ema.update(prob))

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
            fused=result.fused_logit,
            scores=result.scores,
            dropped=False,
            audio_quality=flag,
            window_index=self.windows_scored,
            t0=t0,
            expert_probabilities=expert_smoothed,
            lr_probability=lr_smoothed,
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
            expert_probabilities={},
            lr_probability=None,
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
        expert_probabilities: dict[str, float],
        lr_probability: float | None,
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
            expert_probabilities=expert_probabilities,
            lr_probability=lr_probability,
            expert_calibrators=self.expert_calibrators,
            policy=self.policy,
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
            "scores": {},
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
