import logging
import torch
import numpy as np

logger = logging.getLogger("vad")

_vad_model = None
_get_speech_timestamps = None

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
            logger.info("[vad] Silero VAD loaded.")
        except Exception as e:
            logger.error(f"[vad] Failed to load Silero VAD: {e}")
            raise
    return _vad_model, _get_speech_timestamps

def has_speech(audio_chunk: np.ndarray, sample_rate: int = 16000, threshold: float = 0.5) -> bool:
    """
    Checks if a given audio chunk contains speech using Silero VAD.
    audio_chunk: 1D numpy array of floats (typically -1.0 to 1.0)
    """
    model, get_ts = get_vad_model()
    
    # Silero VAD expects a torch tensor
    tensor = torch.from_numpy(audio_chunk)
    
    # get_speech_timestamps returns a list of dicts [{'start': 123, 'end': 456}, ...]
    with torch.no_grad():
        timestamps = get_ts(
            tensor,
            model,
            sampling_rate=sample_rate,
            threshold=threshold
        )
    
    return len(timestamps) > 0
