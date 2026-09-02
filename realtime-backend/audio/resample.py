from __future__ import annotations

import logging
from math import gcd

import numpy as np

logger = logging.getLogger("realtime_backend.audio")


def to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 0:
        raise ValueError(f"channels must be >= 1, got {channels}")
    if samples.ndim == 1:
        if channels == 1:
            return samples.astype(np.float32, copy=False)
        if samples.size % channels != 0:
            raise ValueError(
                f"interleaved PCM length {samples.size} is not divisible by channels={channels}"
            )
        framed = samples.reshape(-1, channels)
        return framed.mean(axis=1).astype(np.float32)
    if samples.ndim == 2:
        if samples.shape[1] == 1:
            return samples[:, 0].astype(np.float32, copy=False)
        return samples.mean(axis=1).astype(np.float32)
    raise ValueError(f"audio must be 1D or 2D, got shape {samples.shape}")


def _linear_resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Naive linear interpolation. No anti-aliasing — fallback only."""
    duration = audio.size / float(source_rate)
    n_out = max(1, int(round(duration * target_rate)))
    src_t = np.linspace(0.0, duration, num=audio.size, endpoint=False, dtype=np.float64)
    dst_t = np.linspace(0.0, duration, num=n_out, endpoint=False, dtype=np.float64)
    return np.interp(dst_t, src_t, audio).astype(np.float32)


def to_target_rate(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> tuple[np.ndarray, bool]:
    """Resample mono float32 audio to target_rate. Returns (audio, did_resample).

    Uses SciPy's polyphase resampler (``resample_poly``), which applies an FIR
    anti-aliasing filter — important when downsampling (e.g. 48 kHz -> 16 kHz),
    where naive linear interpolation folds >8 kHz energy back into the band as
    aliasing artifacts the detector never saw in training. Falls back to linear
    interpolation only if SciPy is unavailable.
    """
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError(f"invalid sample rates source={source_rate} target={target_rate}")
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate == target_rate:
        return audio, False
    if audio.size == 0:
        return audio, True

    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(source_rate), int(target_rate))
        up = int(target_rate) // g
        down = int(source_rate) // g
        resampled = resample_poly(audio, up, down).astype(np.float32)
    except ImportError:
        logger.warning("scipy unavailable; falling back to linear resample (no anti-aliasing)")
        resampled = _linear_resample(audio, source_rate, target_rate)

    logger.debug(
        "Resampled incoming audio %d Hz -> %d Hz (%d samples -> %d samples)",
        source_rate,
        target_rate,
        audio.size,
        resampled.size,
    )
    return resampled, True


class StreamingPolyphaseResampler:
    """Stateful anti-aliased resampler for the PCM frames of one connection.

    ``to_target_rate`` is correct for a complete file. Calling it once per
    incoming frame is not. The browser's AudioWorklet hands us 128 samples at a
    time, and running a 61-tap anti-aliasing FIR over 128-sample islands
    restarts the filter — and re-rounds the output length — every 2.7 ms. At
    48 kHz -> 16 kHz that measured ~138% RMS error against a correctly resampled
    stream, plus +0.78% sample-rate drift (16125 samples emitted per second of
    input instead of 16000).

    This keeps the FIR state and the decimation phase across calls, so a stream
    delivered in 128-sample frames produces the same samples as the same audio
    resampled in one shot, apart from a fixed group-delay lag held back until
    more input arrives. The filter is the Kaiser design
    ``scipy.signal.resample_poly`` uses by default, so the anti-aliasing the
    offline path relies on is preserved: naive linear interpolation would fold
    everything above the target Nyquist back into the band as artifacts no
    detector saw in training.

    One instance per connection. Call :meth:`reset` on a stream discontinuity;
    never share an instance between streams or sample rates.
    """

    def __init__(self, source_rate: int, target_rate: int):
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError(f"invalid sample rates source={source_rate} target={target_rate}")
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        divisor = gcd(self.source_rate, self.target_rate)
        self.up = self.target_rate // divisor
        self.down = self.source_rate // divisor
        self._h: np.ndarray | None = None
        self._half_len = 0
        if self.did_resample:
            try:
                from scipy.signal import firwin
            except ImportError as exc:  # pragma: no cover - scipy is a hard requirement
                raise ImportError(
                    "scipy is required to resample a live stream without aliasing; "
                    "install scipy>=1.10.0 (see realtime-backend/requirements.txt)"
                ) from exc
            max_rate = max(self.up, self.down)
            self._half_len = 10 * max_rate
            self._h = firwin(
                2 * self._half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0)
            ) * self.up
        self.reset()

    @property
    def did_resample(self) -> bool:
        return self.source_rate != self.target_rate

    def reset(self) -> None:
        """Drop all filter state. Use on a stream gap, not mid-stream."""
        self._zi = None if self._h is None else np.zeros(self._h.size - 1, dtype=np.float64)
        self._phase = 0
        self._warmup = self._half_len

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Resample one frame, continuing from where the last frame ended."""
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not self.did_resample or audio.size == 0:
            return audio
        if not np.isfinite(audio).all():
            raise ValueError("audio contains non-finite samples")

        from scipy.signal import lfilter

        if self.up == 1:
            upsampled = audio.astype(np.float64)
        else:
            upsampled = np.zeros(audio.size * self.up, dtype=np.float64)
            upsampled[:: self.up] = audio

        filtered, self._zi = lfilter(self._h, 1.0, upsampled, zi=self._zi)

        # Drop the filter's group delay once, so output sample j lands at the
        # instant to_target_rate() would have placed it.
        if self._warmup:
            drop = min(self._warmup, filtered.size)
            filtered = filtered[drop:]
            self._warmup -= drop

        out = filtered[self._phase :: self.down]
        # Carry the decimation position into the next frame instead of
        # restarting at 0, which is what caused the per-frame drift.
        self._phase += out.size * self.down - filtered.size
        return out.astype(np.float32)
