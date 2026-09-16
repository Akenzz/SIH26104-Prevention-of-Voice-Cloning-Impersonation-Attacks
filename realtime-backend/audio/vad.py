import logging
import os
import torch
import numpy as np

logger = logging.getLogger("vad")

_vad_model = None
_get_speech_timestamps = None

# Default VAD threshold for Silero VAD.  The published default (0.5) is tuned
# for close-mic studio speech.  When the live monitor captures audio played
# from a loudspeaker through a room (the "play spoof near the mic" scenario),
# reverberation and distance reduce the instantaneous energy peak, causing
# Silero to classify the window as "no speech" and silently drop it before the
# anti-spoof experts ever see it.  Lowering to 0.3 passes room-reverberant
# audio through to the models while still rejecting genuine silence.
# Override with VAD_THRESHOLD=<float> if the deployment environment is noisier.
_DEFAULT_VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.3"))


def get_vad_model():
    global _vad_model, _get_speech_timestamps
    if _vad_model is None:
        logger.info("[vad] Loading Silero VAD model...")
        try:
            # force_reload=False allows using the cached version
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            _vad_model = model
            _get_speech_timestamps = utils[0]
            logger.info(
                "[vad] Silero VAD loaded. Default speech threshold: %.2f "
                "(override with VAD_THRESHOLD env var).",
                _DEFAULT_VAD_THRESHOLD,
            )
        except Exception as e:
            logger.error(f"[vad] Failed to load Silero VAD: {e}")
            raise
    return _vad_model, _get_speech_timestamps


def has_speech(
    audio_chunk: np.ndarray,
    sample_rate: int = 16000,
    threshold: float | None = None,
) -> bool:
    """Check if a given audio chunk contains speech using Silero VAD.

    Args:
        audio_chunk: 1-D numpy array of floats (typically -1.0 to 1.0).
        sample_rate: Sample rate of ``audio_chunk``.
        threshold: Silero confidence threshold in [0, 1].  Lower = more
            permissive (passes more frames, including room-reverberant
            loudspeaker audio).  Defaults to ``VAD_THRESHOLD`` env var
            (0.3 if unset).  Pass an explicit value to override per-call.
    """
    if threshold is None:
        threshold = _DEFAULT_VAD_THRESHOLD

    model, get_ts = get_vad_model()

    # Silero VAD expects a torch tensor
    tensor = torch.from_numpy(audio_chunk)

    # get_speech_timestamps returns a list of dicts [{'start': 123, 'end': 456}, ...]
    with torch.no_grad():
        timestamps = get_ts(
            tensor,
            model,
            sampling_rate=sample_rate,
            threshold=threshold,
        )

    return len(timestamps) > 0
