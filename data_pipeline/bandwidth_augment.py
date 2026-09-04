"""Kill the resampler-bandwidth shortcut by randomizing the top of the band.

WHY THIS EXISTS
---------------
The hybrid corpus taught the LFCC models "spectral hole just under Nyquist =>
spoof". Cause: preprocess_parity.load_mono_16k resamples with librosa/soxr, which
brickwalls 7.9-8 kHz to about -61 dB, and the native sample rates are split by
label -- spoof 52.5% at 22050 Hz (so it WAS resampled -> hole), bonafide 75.8%
already at 16 kHz (so it was NOT -> full band). The realtime backend resamples
with scipy resample_poly, which leaves that band only ~3 dB down, so live audio
never has the hole and in-train generators read bonafide.
Measured: the same Edge-TTS/ml clip scores -9.16 (bonafide) via scipy and +20.12
(spoof) via librosa; brickwalling the scipy version at 7800 Hz alone flips it to
+7.68. See memory/lfcc-resampler-bandwidth-shortcut.md.

THE FIX
-------
Apply a RANDOM bandwidth/resampler fingerprint to every training window with the
same distribution for bonafide and spoof. Once both classes carry every possible
top-band signature, the band cannot predict the label and the LCNN has to use
cues that survive a resample -- which is what the deployment path gives it.

Applied per-sample at train time only (never dev/eval), AFTER cropping to the
window so the cost is bounded, and identically per class -- the caller must NOT
condition the probability on the label.
"""

from __future__ import annotations

import numpy as np

TARGET_SR = 16000

# Intermediate rates a real clip may have passed through. 16000 = "no round trip",
# kept in the pool so some samples stay full-band.
_ROUND_TRIP_RATES = (16000, 22050, 24000, 32000, 44100, 48000)

# Resampler fingerprints. Each has a distinctly different rolloff near Nyquist:
#   soxr_hq / soxr_vhq -> brickwall, ~-60 dB at 7.9-8 kHz (what TRAINING did)
#   scipy resample_poly -> soft, only ~-3 dB there (what the BACKEND does)
#   polyphase -> intermediate
# librosa's kaiser_fast/kaiser_best are deliberately NOT here: they need the
# optional `resampy` package, which is absent on this box, so including them
# meant ~2 of every 5 round trips silently no-op'd through the except branch and
# the augmentation was weaker than it looked. Verify availability before adding.
_METHODS = ("soxr_hq", "soxr_vhq", "scipy_poly", "polyphase")


def _resample(x: np.ndarray, sr_from: int, sr_to: int, method: str) -> np.ndarray:
    if sr_from == sr_to:
        return x.astype(np.float32, copy=False)
    if method == "scipy_poly":
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr_from), int(sr_to))
        return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)
    import librosa

    res_type = {
        "soxr_hq": "soxr_hq",
        "soxr_vhq": "soxr_vhq",
        "kaiser_fast": "kaiser_fast",
        "polyphase": "polyphase",
    }[method]
    return librosa.resample(
        x, orig_sr=sr_from, target_sr=sr_to, res_type=res_type
    ).astype(np.float32)


def _lowpass(x: np.ndarray, cutoff_hz: float, sr: int = TARGET_SR) -> np.ndarray:
    """Zero-phase brickwall via rFFT. Cheap and exactly controls the cutoff."""
    if cutoff_hz >= sr / 2:
        return x
    n = len(x)
    if n < 32:
        return x
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec[freqs >= cutoff_hz] = 0.0
    return np.fft.irfft(spec, n=n).astype(np.float32)


# Resamplers that BRICKWALL the top band (stamp the hole training already had).
_HOLE_METHODS = ("soxr_hq", "soxr_vhq")
# Resamplers that leave the top band nearly intact (what the backend does).
# scipy_poly and polyphase measure identically (-5.7 dB at 7.9-8 kHz); both are
# kept so the model sees more than one no-hole filter shape.
_KEEP_METHODS = ("scipy_poly", "polyphase")


class BandwidthAugment:
    """Randomize resampler fingerprint + audio bandwidth, label-independently.

    Draws ONE mutually-exclusive treatment per window, so the marginal share of
    full-band vs holed audio is a knob rather than an emergent product of two
    independent coins. That matters: an earlier version chained
    p_round_trip=0.6 with p_lowpass=0.4 and drove full-band spoof windows from
    25% down to ~4%, which fixes the shortcut by removing the very case the
    deployment path produces (backend scipy audio is full-band). Measured on 300
    train windows, label/top-band correlation -0.204 unaugmented -> -0.045 there,
    but at that cost.

    Treatments and why each is in the pool:
      "keep"       leave the band as-is, or round-trip through a no-hole
                   resampler -> full-band examples of BOTH classes, matching the
                   live scipy path.
      "hole"       soxr round trip and/or a lowpass in the near-Nyquist range ->
                   reproduces the training-corpus artifact, now on both classes.
      "narrowband" lowpass well below Nyquist (telephony-ish) -> covers cutoffs
                   no resampler round trip produces and stops the model keying on
                   one precise band edge.

    NOTHING here may read the label. Conditioning the draw on the label would
    re-create the shortcut this exists to remove.
    """

    def __init__(
        self,
        p_keep: float = 0.45,
        p_hole: float = 0.45,
        p_narrowband: float = 0.10,
        hole_cutoff_range_hz: tuple[float, float] = (7000.0, 7950.0),
        narrowband_cutoff_range_hz: tuple[float, float] = (3500.0, 6500.0),
        sample_rate: int = TARGET_SR,
        seed: int | None = None,
    ):
        total = float(p_keep) + float(p_hole) + float(p_narrowband)
        if total <= 0:
            raise ValueError("treatment probabilities must sum to > 0")
        self._p = np.array([p_keep, p_hole, p_narrowband], dtype=np.float64) / total
        self.hole_cutoff_range_hz = hole_cutoff_range_hz
        self.narrowband_cutoff_range_hz = narrowband_cutoff_range_hz
        self.sample_rate = int(sample_rate)
        self._rng = np.random.default_rng(seed)

    def _round_trip(self, x: np.ndarray, methods: tuple[str, ...]) -> np.ndarray:
        rate = int(self._rng.choice(_ROUND_TRIP_RATES))
        if rate == self.sample_rate:
            return x
        up_m = str(self._rng.choice(methods))
        down_m = str(self._rng.choice(methods))
        try:
            y = _resample(x, self.sample_rate, rate, up_m)
            return _resample(y, rate, self.sample_rate, down_m)
        except Exception:
            return x  # never let augmentation kill a training step

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Augment one mono float32 window. Length is preserved."""
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        n_in = x.size
        if n_in < 64:
            return x

        treatment = int(self._rng.choice(3, p=self._p))

        if treatment == 0:  # keep the band
            # Half the time still pay a round trip, so "full band" is not
            # perfectly correlated with "never resampled".
            if self._rng.random() < 0.5:
                x = self._round_trip(x, _KEEP_METHODS)
        elif treatment == 1:  # stamp a near-Nyquist hole
            if self._rng.random() < 0.5:
                x = self._round_trip(x, _HOLE_METHODS)
            else:
                lo, hi = self.hole_cutoff_range_hz
                try:
                    x = _lowpass(x, float(self._rng.uniform(lo, hi)), self.sample_rate)
                except Exception:
                    pass
        else:  # narrowband
            lo, hi = self.narrowband_cutoff_range_hz
            try:
                x = _lowpass(x, float(self._rng.uniform(lo, hi)), self.sample_rate)
            except Exception:
                pass

        # Length must stay exactly the window the model expects.
        if x.size < n_in:
            x = np.pad(x, (0, n_in - x.size))
        elif x.size > n_in:
            x = x[:n_in]
        return x.astype(np.float32, copy=False)
