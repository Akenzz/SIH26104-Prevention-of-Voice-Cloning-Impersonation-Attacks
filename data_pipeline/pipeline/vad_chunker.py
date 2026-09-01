"""
vad_chunker.py — Silero VAD-based segmentation of long audio files.

Strategy:
  - Load Silero VAD model once (shared across calls).
  - Get speech segment timestamps.
  - Merge nearby segments (< MERGE_GAP_S apart).
  - Group merged segments into chunks targeting 4–10 s.
  - Discard any resulting chunk < MIN_CHUNK_S.
  - Write chunks as {base_id}_chunk{N:02d}.wav.
  - Skip if all chunk files already exist (resumable).

Returns list of (chunk_path, duration_s) tuples.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger("pipeline.vad_chunker")

TARGET_SR    = 16_000
MIN_CHUNK_S  = 1.5
MAX_CHUNK_S  = 12.0
TARGET_MIN_S = 4.0
TARGET_MAX_S = 10.0
MERGE_GAP_S  = 0.3

_vad_model = None
_vad_utils = None


def _get_vad():
    global _vad_model, _vad_utils
    if _vad_model is None:
        import torch
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        _vad_model.eval()
    return _vad_model, _vad_utils


def chunk_file(
    audio: np.ndarray,
    sr: int,
    dst_dir: Path,
    base_id: str,
    skip_if_exists: bool = True,
) -> List[Tuple[Path, float]]:
    """
    Segment `audio` using Silero VAD and write chunks to `dst_dir`.
    Returns list of (path, duration_s) for each written chunk.
    If the file is already chunked (all expected files exist), returns their info.
    """
    import torch

    dst_dir.mkdir(parents=True, exist_ok=True)

    # If already chunked, detect and return
    if skip_if_exists:
        existing = sorted(dst_dir.glob(f"{base_id}_chunk*.wav"))
        if existing:
            return [(p, sf.info(str(p)).duration) for p in existing]

    # Ensure 16kHz for VAD
    if sr != TARGET_SR:
        import librosa  # type: ignore
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    model, utils = _get_vad()
    get_speech_timestamps = utils[0]

    tensor = torch.from_numpy(audio).float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)  # [1, T]

    try:
        speech_timestamps = get_speech_timestamps(
            tensor.squeeze(0),
            model,
            sampling_rate=sr,
            min_speech_duration_ms=300,
            min_silence_duration_ms=200,
            threshold=0.5,
        )
    except Exception as e:
        logger.warning("VAD failed for %s: %s. Returning empty.", base_id, e)
        return []

    if not speech_timestamps:
        # No speech detected — treat whole file as one chunk if it fits
        duration = len(audio) / sr
        if MIN_CHUNK_S <= duration <= MAX_CHUNK_S:
            out_path = dst_dir / f"{base_id}_chunk00.wav"
            sf.write(str(out_path), audio, sr, subtype="PCM_16")
            return [(out_path, duration)]
        return []

    # Convert to (start_s, end_s) list
    segments = [
        (ts["start"] / sr, ts["end"] / sr) for ts in speech_timestamps
    ]

    # Merge nearby segments
    merged = _merge_segments(segments, MERGE_GAP_S)

    # Group into TARGET_MIN_S – TARGET_MAX_S chunks
    chunks_ranges = _group_into_chunks(merged, TARGET_MIN_S, TARGET_MAX_S, MAX_CHUNK_S)

    results = []
    for i, (start_s, end_s) in enumerate(chunks_ranges):
        duration = end_s - start_s
        if duration < MIN_CHUNK_S:
            continue
        start_sample = int(start_s * sr)
        end_sample   = int(end_s   * sr)
        chunk_audio  = audio[start_sample:end_sample]
        out_path     = dst_dir / f"{base_id}_chunk{i:02d}.wav"
        sf.write(str(out_path), chunk_audio, sr, subtype="PCM_16")
        results.append((out_path, duration))

    return results


def _merge_segments(segments: list, gap_s: float) -> list:
    """Merge adjacent segments separated by less than gap_s seconds."""
    if not segments:
        return []
    merged = [list(segments[0])]
    for start, end in segments[1:]:
        if start - merged[-1][1] < gap_s:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [tuple(s) for s in merged]


def _group_into_chunks(
    segments: list,
    target_min: float,
    target_max: float,
    hard_max: float,
) -> list:
    """
    Greedily group segments into chunks targeting [target_min, target_max] seconds.
    A segment that's already in [target_min, target_max] is kept as-is.
    """
    chunks = []
    current_start = None
    current_end   = None

    for seg_start, seg_end in segments:
        seg_dur = seg_end - seg_start

        if current_start is None:
            current_start = seg_start
            current_end   = seg_end
        else:
            potential_dur = seg_end - current_start
            if potential_dur <= target_max:
                current_end = seg_end
            else:
                # Flush current chunk
                chunks.append((current_start, current_end))
                current_start = seg_start
                current_end   = seg_end

    if current_start is not None:
        chunks.append((current_start, current_end))

    # Split any chunk that is still > hard_max by time
    final = []
    for start, end in chunks:
        dur = end - start
        if dur <= hard_max:
            final.append((start, end))
        else:
            # Blind split at target_max boundaries
            t = start
            while t < end:
                final.append((t, min(t + target_max, end)))
                t += target_max

    return final
