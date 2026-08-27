"""
generate_synthetic_data.py — Creates placeholder audio files and a manifest CSV
so the full pipeline can run end-to-end before your real dataset is ready.

Generated files:
    data/audio/*.wav     — short synthetic wav files (random noise or pure tones)
    data/manifest.csv    — manifest in the exact schema expected by dataset.py

# TODO: Delete this file and replace data/manifest.csv with your real manifest
#       (e.g. pointing at ASVspoof2019-LA audio) when your real data is ready.
#       No changes to any other source file are required.
"""

import os
import csv
import random
import math

import numpy as np
import soundfile as sf

# ─── Config ────────────────────────────────────────────────────────────────────
# Paths are resolved relative to THIS script's directory (expert1/),
# so the script works whether you run it from the project root or from expert1/.
_HERE        = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR    = os.path.join(_HERE, "data", "audio")
MANIFEST_CSV = os.path.join(_HERE, "data", "manifest.csv")
SAMPLE_RATE  = 16_000
DURATION_S   = 4          # seconds per clip (matches WINDOW_SECONDS in dataset.py)
NUM_SAMPLES  = 60         # total synthetic clips (20 train bonafide + 20 train spoof
                          #                        +  5 dev each +  5 test each)
SEED         = 42

random.seed(SEED)
np.random.seed(SEED)

os.makedirs(AUDIO_DIR, exist_ok=True)

# ── Split / label distribution ──────────────────────────────────────────────────
DISTRIBUTION = [
    # (split, label, count)
    ("train", "bonafide", 20),
    ("train", "spoof",    20),
    ("dev",   "bonafide",  5),
    ("dev",   "spoof",     5),
    ("test",  "bonafide",  5),
    ("test",  "spoof",     5),
]

N_SAMPLES = SAMPLE_RATE * DURATION_S   # 64 000 samples per file


def make_bonafide_audio() -> np.ndarray:
    """
    Simulate a 'real' speech-like signal: a mixture of a few sine waves
    with amplitude modulation (simple voice-like envelope).
    """
    t = np.linspace(0, DURATION_S, N_SAMPLES, endpoint=False)
    freq = random.uniform(80, 300)        # rough fundamental frequency
    audio  = 0.4 * np.sin(2 * math.pi * freq * t)
    audio += 0.2 * np.sin(2 * math.pi * freq * 2 * t)
    audio += 0.1 * np.sin(2 * math.pi * freq * 3 * t)
    # Amplitude modulation to simulate syllable rhythm
    mod = 0.5 * (1 + np.sin(2 * math.pi * 4 * t))
    audio = audio * mod
    # Add a little noise
    audio += 0.02 * np.random.randn(N_SAMPLES)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)   # normalise to [-1, 1]
    return audio.astype(np.float32)


def make_spoof_audio() -> np.ndarray:
    """
    Simulate an 'AI-generated' signal: band-limited white noise
    (spectral flatness is a common artefact of early TTS/VC systems).
    """
    audio = np.random.randn(N_SAMPLES)
    # Band-limit via a very simple moving average (crude low-pass)
    kernel = np.ones(32) / 32
    audio  = np.convolve(audio, kernel, mode="same")
    audio  = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio.astype(np.float32)


# ── Generate files and build manifest rows ──────────────────────────────────────
MANIFEST_COLUMNS = [
    "path", "label", "split", "source_dataset", "speaker_id",
    "utterance_id", "generator_id", "language", "codec",
    "duration_s", "license", "consent",
]

rows = []
uid  = 0

for split, label, count in DISTRIBUTION:
    for i in range(count):
        uid += 1
        filename = f"synth_{uid:04d}_{label}.wav"
        filepath = os.path.join(AUDIO_DIR, filename)

        # Generate audio
        if label == "bonafide":
            audio = make_bonafide_audio()
            generator_id = "none"           # no TTS/VC system
        else:
            audio = make_spoof_audio()
            generator_id = "synthetic_noise_v1"

        # Save as 16 kHz mono WAV
        sf.write(filepath, audio, SAMPLE_RATE, subtype="PCM_16")

        rows.append({
            "path"           : filepath,
            "label"          : label,
            "split"          : split,
            "source_dataset" : "synthetic_placeholder",
            "speaker_id"     : f"spk_{uid:04d}",
            "utterance_id"   : f"utt_{uid:04d}",
            "generator_id"   : generator_id,
            "language"       : "en",
            "codec"          : "pcm_16",
            "duration_s"     : DURATION_S,
            "license"        : "cc0",
            "consent"        : "n/a",
        })

# ── Write manifest CSV ─────────────────────────────────────────────────────────
with open(MANIFEST_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

print(f"[generate] Created {len(rows)} audio files in '{AUDIO_DIR}/'")
print(f"[generate] Manifest written to '{MANIFEST_CSV}'")
print()
print("Split breakdown:")
for split, label, count in DISTRIBUTION:
    print(f"  {split:5s}  {label:9s}  {count} clips")
print()
print("# TODO: When your real dataset is ready, replace data/manifest.csv with")
print("#       a manifest pointing at your real audio files. No other changes needed.")
