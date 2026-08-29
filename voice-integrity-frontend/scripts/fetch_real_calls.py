"""Build real call-audio test clips from ASVspoof2019 LA (the detector's own domain).

The synthetic tones in test-samples/ only exercise the plumbing — they are not
speech, so any verdict from them is meaningless. This pulls genuine human speech
and real voice-conversion spoofs of *the same speaker* from the ASVspoof2019 LA
development set and stitches each into a call-length 16 kHz mono WAV, so the
console has something honest to score.

    python scripts/fetch_real_calls.py                 # defaults: LA_0076, A06, 24 s
    python scripts/fetch_real_calls.py --attack A01 --seconds 40

Output lands in test-samples/real-calls/ alongside a MANIFEST.txt recording
exactly which utterances went into each file.

IMPORTANT — read before quoting any number these produce: the clips come from
the LA *dev* partition, which is the partition the shipped calibrator
(platt-lfcc-asvspoof19dev-v1) was fitted on. Scores here are in-domain and
flattering. They demonstrate the system end to end; they are not a held-out
accuracy measurement. For that you need the LA *eval* partition.

Requires: huggingface_hub, soundfile, numpy.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = "Nemez1z/asvspoof-2019-la"
PROTOCOL = "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
FLAC_DIR = "ASVspoof2019_LA_dev/flac"
SR = 16000  # ASVspoof LA is already 16 kHz; assert rather than resample

ATTACK_NAMES = {
    "A01": "neural TTS (wavenet vocoder)",
    "A02": "neural TTS (world vocoder)",
    "A03": "neural TTS (feed-forward)",
    "A04": "waveform-concatenation TTS",
    "A05": "voice conversion (neural)",
    "A06": "voice conversion (spectral filtering)",
}


def load_protocol(hf_hub_download, api):
    """Return {speaker: {'bonafide'|attack_id: [utt_id, ...]}} for downloadable clips."""
    path = hf_hub_download(REPO, PROTOCOL, repo_type="dataset")
    available = {
        f.split("/")[-1][: -len(".flac")]
        for f in api.list_repo_files(REPO, repo_type="dataset")
        if f.endswith(".flac")
    }
    table = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            cols = line.split()
            if len(cols) < 5 or cols[1] not in available:
                continue
            speaker, utt, _, attack, label = cols[0], cols[1], cols[2], cols[3], cols[4]
            key = "bonafide" if label == "bonafide" else attack
            table[speaker][key].append(utt)
    return table


def fetch_clip(hf_hub_download, sf, utt: str) -> np.ndarray:
    path = hf_hub_download(REPO, f"{FLAC_DIR}/{utt}.flac", repo_type="dataset")
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if rate != SR:
        raise SystemExit(f"{utt}: expected {SR} Hz, got {rate} Hz")
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def build_call(hf_hub_download, sf, utts: list[str], seconds: float, gap_ms: int = 220):
    """Concatenate utterances up to `seconds`, with a short gap between turns."""
    gap = np.zeros(int(SR * gap_ms / 1000.0), dtype=np.float32)
    target = int(SR * seconds)
    parts: list[np.ndarray] = []
    used: list[str] = []
    total = 0
    for utt in utts:
        clip = fetch_clip(hf_hub_download, sf, utt)
        parts.extend([clip, gap])
        used.append(f"{utt} ({clip.size / SR:.1f}s)")
        total += clip.size + gap.size
        if total >= target:
            break
    if not parts:
        raise SystemExit("no clips fetched")
    return np.concatenate(parts)[:target], used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", default="LA_0076", help="LA dev speaker id")
    ap.add_argument("--attack", default="A06", choices=sorted(ATTACK_NAMES), help="spoof system")
    ap.add_argument("--seconds", type=float, default=24.0, help="length of each built call")
    ap.add_argument("--singles", type=int, default=2, help="also save N single utterances per class")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "test-samples" / "real-calls",
    )
    args = ap.parse_args()

    try:
        import soundfile as sf
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit(f"missing dependency: {exc.name} — pip install huggingface_hub soundfile") from exc

    print(f"reading the LA dev protocol from {REPO}...")
    table = load_protocol(hf_hub_download, HfApi())
    if args.speaker not in table:
        raise SystemExit(f"unknown speaker {args.speaker}; try one of {sorted(table)[:6]}")
    lists = table[args.speaker]
    for key in ("bonafide", args.attack):
        if not lists.get(key):
            raise SystemExit(f"{args.speaker} has no {key} clips in this mirror")

    rng = random.Random(args.seed)
    picks = {k: rng.sample(lists[k], len(lists[k])) for k in ("bonafide", args.attack)}

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = [
        f"ASVspoof2019 LA dev · speaker {args.speaker} · spoof system {args.attack} "
        f"({ATTACK_NAMES[args.attack]})",
        "Dev partition = the partition the shipped calibrator was fitted on. In-domain, flattering.",
        "",
    ]

    jobs = [
        ("bonafide", f"call_genuine_{args.speaker}.wav", "genuine caller — expect LOW risk"),
        (args.attack, f"call_cloned_{args.attack}_{args.speaker}.wav", f"{args.attack} clone — expect HIGH risk"),
    ]
    for key, filename, note in jobs:
        audio, used = build_call(hf_hub_download, sf, picks[key], args.seconds)
        dest = args.out / filename
        sf.write(dest, audio, SR, subtype="PCM_16")
        print(f"  wrote {dest.name}  {audio.size / SR:.1f}s  ({note})")
        manifest += [f"{filename} — {note}", *(f"    {u}" for u in used), ""]

    for key, tag in (("bonafide", "genuine"), (args.attack, f"cloned_{args.attack}")):
        for utt in picks[key][: args.singles]:
            clip = fetch_clip(hf_hub_download, sf, utt)
            dest = args.out / f"single_{tag}_{utt}.wav"
            sf.write(dest, clip, SR, subtype="PCM_16")
            print(f"  wrote {dest.name}  {clip.size / SR:.1f}s")
            manifest.append(f"{dest.name} — single utterance {utt} ({tag})")

    (args.out / "MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"\ndone -> {args.out}")
    print("Clips under 4 s cannot fill a window; use the call_* files for a real timeline.")


if __name__ == "__main__":
    sys.exit(main())
