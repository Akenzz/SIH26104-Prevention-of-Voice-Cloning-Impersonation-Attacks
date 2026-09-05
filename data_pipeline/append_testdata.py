#!/usr/bin/env python3
"""
Append testdata clips to train.csv so WavLM learns from real-world
domain examples before the next fine-tuning run.

Safe rules:
  - bonafied5.wav is EXCLUDED (24kHz, scores 86%/80% spoof → likely mislabeled)
  - Only clear-cut bonafide / clear-cut spoof files are added
  - We never modify the original train.csv; we write a new v2_extended/train.csv
"""
import csv, os, shutil
from pathlib import Path

TESTDATA  = Path("/home/akenzz/sih/project/testdata")
ORIGINAL  = Path("/media/akenzz/D/DataSet_processed/v2")
EXTENDED  = Path("/media/akenzz/D/DataSet_processed/v2_extended")
EXTENDED.mkdir(parents=True, exist_ok=True)

# ── Files to add ─────────────────────────────────────────────────────────────
# Format: (filename, label, generator, language)
# bonafied5.wav intentionally EXCLUDED (24kHz + both models say >80% spoof)
NEW_FILES = [
    # --- BONAFIDE ---
    ("bonafied1.mp3",  "bonafide", "bonafide_testdata", "en"),   # WavLM false-positive to fix
    ("bonafied2.wav",  "bonafide", "bonafide_testdata", "en"),
    ("bonafied3.wav",  "bonafide", "bonafide_testdata", "en"),
    ("bonafied4.wav",  "bonafide", "bonafide_testdata", "en"),
    ("bonafied6.wav",  "bonafide", "bonafide_testdata", "en"),

    # --- SPOOF (WavLM missed these) ---
    ("sarosh_chatterbox_spoof.wav",  "spoof", "Chatterbox",         "en"),
    ("sarosh_fireredtts.wav",        "spoof", "FireRedTTS-2.0",     "en"),
    ("sarosh_omni.wav",              "spoof", "Qwen2.5-Omni",       "en"),
    ("sarosh_spoof.wav",             "spoof", "spoof_testdata",     "en"),
    ("sarosh_styletts.wav",          "spoof", "styletts2",          "en"),
    ("me_spoofed_qwen.wav",          "spoof", "Qwen3-TTS-CustomVoice", "en"),
    ("my_spoof_eng.wav",             "spoof", "spoof_testdata",     "en"),
    ("sudhu_eng_spoof.wav",          "spoof", "spoof_testdata",     "en"),
    ("SAROSH_SOPRO_V2_SPOOF.wav",    "spoof", "spoof_testdata",     "en"),
    # Note: tmp7y7ms46r.wav, spk_1788276914.wav, temp_output.mp3 are excluded
    # (unclear origin — could be bonafide or spoof)
]

def process_split(split_name):
    src  = ORIGINAL / f"{split_name}.csv"
    dst  = EXTENDED / f"{split_name}.csv"
    shutil.copy2(src, dst)
    print(f"Copied {src} → {dst}")

    if split_name != "train":
        return  # only inject into train

    appended = 0
    with open(dst, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for fname, label, generator, lang in NEW_FILES:
            path = TESTDATA / fname
            if not path.exists():
                print(f"  ⚠ MISSING: {fname} — skipping")
                continue
            writer.writerow([str(path), label, generator, lang, "train"])
            print(f"  ✓ added {fname} as {label} ({generator})")
            appended += 1

    print(f"\nAppended {appended} new rows to {dst}")

process_split("train")
process_split("val")
print(f"\nDone! New training CSV: {EXTENDED}/train.csv")
print("Val CSV is unchanged (copied as-is).")
