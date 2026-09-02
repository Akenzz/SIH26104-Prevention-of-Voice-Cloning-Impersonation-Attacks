"""Task F honesty benchmark — check narration against the real numbers.

`final.md` asks for a manual check of 5-10 narration outputs against the actual
per-window fields, with ZERO hallucinated evidence. This script automates that
guard:

* It never lets forbidden acoustic-evidence vocabulary through (spectral,
  artifact, prosody, formant, breath, jitter, shimmer, harmonic, pitch, timbre,
  waveform, frequency) — those are never in the allowlisted fields, so any
  appearance is invented. This is a hard failure.
* It confirms `sanitize()` drops everything outside the allowlist, so the LLM
  never sees more than the detector emitted.
* With GROQ_API_KEY set it calls the live model per fixture and prints the
  sentence beside its source JSON for the manual eyeball the spec asks for.

The local (offline) narrator is JavaScript (voice-integrity-frontend/src/lib/
narrator.js) and is grounded by construction; eyeball it live in the UI.

Run from realtime-backend/:  python scripts/check_narration.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from narration import ALLOWED_FIELDS, SYSTEM_PROMPT, sanitize, stream_groq  # noqa: E402

FORBIDDEN = re.compile(
    r"\b(spectral|artifact|prosod|formant|breath|jitter|shimmer|harmonic|"
    r"pitch|timbre|waveform|frequenc)",
    re.IGNORECASE,
)

# Representative windows across every risk state. Each carries an extra
# `secret_field` that sanitize() must strip before anything reaches the LLM.
FIXTURES = [
    {
        "window_index": 2,
        "risk_state": "collecting",
        "smoothed_probability": 0.12,
        "raw_per_expert_scores": {"hybrid": -6.10, "wavlm": -3.20},
        "audio_quality": None,
        "recommended_action": "Keep the call going; not enough audio yet to score.",
        "secret_field": "should never reach the model",
    },
    {
        "window_index": 8,
        "risk_state": "low",
        "smoothed_probability": 0.18,
        "raw_per_expert_scores": {"hybrid": -5.02, "wavlm": -2.90},
        "audio_quality": None,
        "recommended_action": "Smoothed spoof risk is low. Continue.",
        "secret_field": "drop me",
    },
    {
        "window_index": 15,
        "risk_state": "uncertain",
        "smoothed_probability": 0.51,
        "raw_per_expert_scores": {"hybrid": 0.40, "wavlm": 1.10},
        "audio_quality": None,
        "recommended_action": "Risk is inconclusive. Pause and use a known callback or MFA.",
        "secret_field": "drop me",
    },
    {
        "window_index": 23,
        "risk_state": "high",
        "smoothed_probability": 0.88,
        "raw_per_expert_scores": {"hybrid": 5.90, "wavlm": 4.20},
        "audio_quality": None,
        "recommended_action": "Smoothed risk is high. Pause and verify through a known callback number.",
        "secret_field": "drop me",
    },
    {
        "window_index": 24,
        "risk_state": "unavailable",
        "smoothed_probability": None,
        "raw_per_expert_scores": {},
        "audio_quality": "clipped",
        "recommended_action": "Audio quality or stream integrity failed.",
        "secret_field": "drop me",
    },
]


def check_forbidden(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in FORBIDDEN.finditer(text)})


def check_sanitize() -> int:
    failures = 0
    for fx in FIXTURES:
        clean = sanitize(fx)
        extra = [k for k in clean if k not in ALLOWED_FIELDS]
        if extra:
            print(f"  FAIL sanitize leaked non-allowlisted keys: {extra}")
            failures += 1
        if "secret_field" in clean:
            print("  FAIL sanitize did not strip secret_field")
            failures += 1
    if failures == 0:
        print("  OK   sanitize() forwards only allowlisted fields")
    return failures


async def run_live(model: str, api_key: str) -> int:
    failures = 0
    for fx in FIXTURES:
        fields = sanitize(fx)
        text = ""
        try:
            async for delta in stream_groq(fields, api_key=api_key, model=model):
                text += delta
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR window {fx['window_index']}: {exc}")
            failures += 1
            continue
        text = text.strip()
        bad = check_forbidden(text)
        print("\n  source :", json.dumps(fields, ensure_ascii=True))
        print("  narrated:", text)
        if bad:
            print(f"  FAIL invented acoustic-evidence words: {bad}")
            failures += 1
        else:
            print("  OK   no invented evidence")
    return failures


def main() -> int:
    print("Task F narration benchmark\n" + "=" * 40)
    print("\n[1] system prompt guardrails")
    for word in ("never invent", "identity", "not explainable ai"):
        ok = word in SYSTEM_PROMPT.lower()
        print(f"  {'OK  ' if ok else 'FAIL'} prompt mentions {word!r}")

    print("\n[2] sanitize() allowlist")
    failures = check_sanitize()

    api_key = os.environ.get("GROQ_API_KEY", "")
    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    print(f"\n[3] live narration ({model})")
    if not api_key:
        print("  SKIP GROQ_API_KEY not set -- offline. Local narrator is grounded")
        print("       by construction; eyeball it live in the UI (Live Monitor).")
    else:
        failures += asyncio.run(run_live(model, api_key))

    print("\n" + "=" * 40)
    print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures} issue(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
