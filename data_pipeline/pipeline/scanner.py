"""
scanner.py — Per-source file scanners that emit raw index rows.

Each scanner is a generator that yields dicts with the fields:
    src_path    : absolute path to the raw source file
    label       : 'bonafide' | 'spoof'
    speaker_id  : str or ''
    language    : str (e.g. 'hi', 'en', 'kn' …)
    generator_id: str or 'none'
    source      : dataset name string
    split_hint  : 'train' | 'dev' | 'eval' | 'eval_ood'
    held_out    : bool

Scanners never modify files.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Generator, Dict, Any

logger = logging.getLogger("pipeline.scanner")

Row = Dict[str, Any]

DATASET_ROOT = Path("/media/akenzz/D/DataSet")
AUDIO_EXTS   = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}

# ──────────────────────────────────────────────────────────────────────────────
# Gramvaani / GV_* — bonafide Hindi speech
# ──────────────────────────────────────────────────────────────────────────────

def scan_gramvaani() -> Generator[Row, None, None]:
    """Gramvaani_1000hrData_Part5 — train, nested by speaker folder."""
    root = DATASET_ROOT / "Gramvaani_1000hrData_Part5"
    yield from _scan_flat_bonafide(root, source="gramvaani_1000h", split_hint="train", language="hi")


def scan_gv_train() -> Generator[Row, None, None]:
    """GV_Train_100h — train, files in Audio/."""
    root = DATASET_ROOT / "GV_Train_100h"
    yield from _scan_flat_bonafide(root, source="gv_train_100h", split_hint="train", language="hi")


def scan_gv_eval() -> Generator[Row, None, None]:
    """GV_Eval_3h — dev, files in Audio/."""
    root = DATASET_ROOT / "GV_Eval_3h"
    yield from _scan_flat_bonafide(root, source="gv_eval_3h", split_hint="dev", language="hi")


def _scan_flat_bonafide(
    root: Path, source: str, split_hint: str, language: str
) -> Generator[Row, None, None]:
    for path in root.rglob("*"):
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        # Speaker ID = immediate parent folder name
        speaker_id = path.parent.name
        yield {
            "src_path"    : str(path),
            "label"       : "bonafide",
            "speaker_id"  : speaker_id,
            "language"    : language,
            "generator_id": "none",
            "source"      : source,
            "split_hint"  : split_hint,
            "held_out"    : False,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Hindi (Kathbath)
# ──────────────────────────────────────────────────────────────────────────────

def scan_hindi() -> Generator[Row, None, None]:
    """Hindi (Kathbath) — bonafide Hindi, paired .wav + .json under v1/train/."""
    root = DATASET_ROOT / "Hindi"

    for wav_path in root.rglob("*.wav"):
        json_path = wav_path.with_suffix(".json")
        speaker_id = ""
        language   = "hi"

        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                speaker_id = str(meta.get("speaker_id", meta.get("speaker", "")))
                language   = _iso_lang(meta.get("language", "Hindi"))
            except Exception as e:
                logger.warning("Malformed JSON for %s: %s — using defaults", wav_path.name, e)
        else:
            logger.warning("Missing JSON for %s — using defaults", wav_path.name)

        yield {
            "src_path"    : str(wav_path),
            "label"       : "bonafide",
            "speaker_id"  : speaker_id,
            "language"    : language,
            "generator_id": "none",
            "source"      : "kathbath",
            "split_hint"  : "train",
            "held_out"    : False,
        }


def _iso_lang(name: str) -> str:
    mapping = {
        "hindi": "hi", "english": "en", "kannada": "kn",
        "malayalam": "ml", "marathi": "mr", "tamil": "ta",
    }
    return mapping.get(name.lower(), name.lower()[:2])


# ──────────────────────────────────────────────────────────────────────────────
# LA (ASVspoof 2019)
# ──────────────────────────────────────────────────────────────────────────────

_LA_PROTOCOL_DIR = DATASET_ROOT / "LA" / "ASVspoof2019_LA_cm_protocols"
_LA_PROTOCOL_FILES = {
    "train": _LA_PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt",
    "dev"  : _LA_PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval" : _LA_PROTOCOL_DIR / "ASVspoof2019.LA.cm.eval.trl.txt",
}
_LA_AUDIO_DIRS = {
    "train": DATASET_ROOT / "LA" / "ASVspoof2019_LA_train" / "flac",
    "dev"  : DATASET_ROOT / "LA" / "ASVspoof2019_LA_dev"   / "flac",
    "eval" : DATASET_ROOT / "LA" / "ASVspoof2019_LA_eval"  / "flac",
}


def scan_la() -> Generator[Row, None, None]:
    """LA (ASVspoof 2019) — labels strictly from CM protocol files."""
    for split, proto_path in _LA_PROTOCOL_FILES.items():
        if not proto_path.exists():
            logger.error("Protocol file missing: %s", proto_path)
            continue

        audio_dir = _LA_AUDIO_DIRS[split]
        if not audio_dir.exists():
            logger.error("Audio dir missing: %s", audio_dir)
            continue

        with open(proto_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                speaker_id  = parts[0]
                utt_id      = parts[1]
                attack_id   = parts[3]   # '-' for bonafide
                key         = parts[4]   # 'bonafide' | 'spoof'

                # Resolve audio path (.flac preferred, fallback .wav)
                audio_path = audio_dir / f"{utt_id}.flac"
                if not audio_path.exists():
                    audio_path = audio_dir / f"{utt_id}.wav"
                if not audio_path.exists():
                    logger.warning("Audio not found for %s — skipping", utt_id)
                    continue

                generator_id = "none" if key == "bonafide" else attack_id
                split_hint   = split  # 'train' | 'dev' | 'eval'

                yield {
                    "src_path"    : str(audio_path),
                    "label"       : key,
                    "speaker_id"  : speaker_id,
                    "language"    : "en",
                    "generator_id": generator_id,
                    "source"      : "asvspoof2019_la",
                    "split_hint"  : split_hint,
                    "held_out"    : False,
                }


# ──────────────────────────────────────────────────────────────────────────────
# MLAAD — spoof-only
# ──────────────────────────────────────────────────────────────────────────────

# Two generators fully held out as "unseen generator" OOD test
MLAAD_HELD_OUT_GENERATORS = {"ElevenLabs-v3", "GPT-SoVITS"}

_MLAAD_ROOT = DATASET_ROOT / "MLAAD" / "MLAAD" / "fake"
_HF_VENV    = ".hf-venv"  # ignore this folder entirely


def scan_mlaad() -> Generator[Row, None, None]:
    """MLAAD — spoof-only, organized as fake/<lang>/<generator>/*.wav.
    MLAAD has NO real/bonafide folder — this is spoof-only, documented.
    """
    if not _MLAAD_ROOT.exists():
        logger.error("MLAAD fake root not found: %s", _MLAAD_ROOT)
        return

    for lang_dir in _MLAAD_ROOT.iterdir():
        if not lang_dir.is_dir():
            continue
        language = lang_dir.name  # 'en', 'hi', 'kn', etc.

        for gen_dir in lang_dir.iterdir():
            if not gen_dir.is_dir():
                continue
            generator_id = gen_dir.name

            # Skip the stray huggingface venv that ended up in the dataset folder
            if _HF_VENV in str(gen_dir):
                continue

            is_held_out = generator_id in MLAAD_HELD_OUT_GENERATORS
            split_hint  = "eval_ood" if is_held_out else "train"

            for audio_file in gen_dir.rglob("*"):
                if audio_file.suffix.lower() not in AUDIO_EXTS:
                    continue
                if _HF_VENV in str(audio_file):
                    continue

                yield {
                    "src_path"    : str(audio_file),
                    "label"       : "spoof",
                    "speaker_id"  : "",
                    "language"    : language,
                    "generator_id": generator_id,
                    "source"      : "mlaad",
                    "split_hint"  : split_hint,
                    "held_out"    : is_held_out,
                }


# ──────────────────────────────────────────────────────────────────────────────
# release_in_the_wild — ALL eval_ood
# ──────────────────────────────────────────────────────────────────────────────

_WILD_ROOT    = DATASET_ROOT / "release_in_the_wild"
_WILD_META    = _WILD_ROOT / "meta.csv"


def scan_release_in_the_wild() -> Generator[Row, None, None]:
    """release_in_the_wild — labels from meta.csv. ALL rows are eval_ood."""
    if not _WILD_META.exists():
        logger.error("meta.csv missing for release_in_the_wild")
        return

    import csv
    with open(_WILD_META, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename    = row.get("file", "").strip()
            raw_label   = row.get("label", "").strip().lower()
            speaker_id  = row.get("speaker", "").strip()

            # Normalize "bona-fide" → "bonafide"
            label = "bonafide" if "bona" in raw_label else "spoof"

            audio_path = _WILD_ROOT / filename
            if not audio_path.exists():
                logger.warning("Audio not found: %s — skipping", audio_path)
                continue

            yield {
                "src_path"    : str(audio_path),
                "label"       : label,
                "speaker_id"  : speaker_id,
                "language"    : "en",
                "generator_id": "none" if label == "bonafide" else "unknown_wild",
                "source"      : "release_in_the_wild",
                "split_hint"  : "eval_ood",
                "held_out"    : True,
            }


# ──────────────────────────────────────────────────────────────────────────────
# processed folder — already standardized, fold in as-is
# ──────────────────────────────────────────────────────────────────────────────

_PROCESSED_ROOT = DATASET_ROOT / "processed"


def scan_processed() -> Generator[Row, None, None]:
    """Already-processed folder — infer label from bonafide/spoof subfolder."""
    for audio_path in _PROCESSED_ROOT.rglob("*.wav"):
        parts = audio_path.parts
        label = "bonafide"
        split_hint = "train"

        for part in parts:
            lp = part.lower()
            if lp == "bonafide":
                label = "bonafide"
            elif lp in ("spoof", "fake"):
                label = "spoof"
            if lp == "dev":
                split_hint = "dev"
            elif lp == "eval":
                split_hint = "eval"
            elif lp == "train":
                split_hint = "train"

        yield {
            "src_path"    : str(audio_path),
            "label"       : label,
            "speaker_id"  : "",
            "language"    : "mixed",
            "generator_id": "none" if label == "bonafide" else "processed_synthetic",
            "source"      : "processed_v2",
            "split_hint"  : split_hint,
            "held_out"    : False,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Master list of all scanners
# ──────────────────────────────────────────────────────────────────────────────

ALL_SCANNERS = [
    ("gramvaani_1000h"    , scan_gramvaani),
    ("gv_train_100h"      , scan_gv_train),
    ("gv_eval_3h"         , scan_gv_eval),
    ("kathbath"           , scan_hindi),
    ("asvspoof2019_la"    , scan_la),
    ("mlaad"              , scan_mlaad),
    ("release_in_the_wild", scan_release_in_the_wild),
    ("processed_v2"       , scan_processed),
]
