"""
compare_clips.py — diff two score_clips.py JSON outputs (before vs after).

Prints a per-clip table of mean-logit and %spoof-window change, and a summary of
how many spoof clips crossed from "missed" (mean logit < 0, i.e. called real) to
"caught" (mean logit > 0). Higher logit = more spoof.

Run from lfcc-detector/:
  python compare_clips.py before_newclips.json after_newclips.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Ground truth for Amogh's clip pack (by filename substring). Everything in the
# folder is a spoof EXCEPT the one mislabeled-by-location real clip; the three
# unnamed files were excluded from training but we still show their score drift.
REAL = {"bonafide_eng_sudhu.mp3"}
EXCLUDED = {"audio(1).wav", "spk_1788276914.wav", "tmp7y7ms46r.wav"}


def truth(name: str) -> str:
    if name in REAL:
        return "real"
    if name in EXCLUDED:
        return "?(excl)"
    return "spoof"


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python compare_clips.py <before.json> <after.json>")
        return 2
    before = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    after = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

    names = [n for n in before if n in after]
    print(f"\n{'clip':30s} {'truth':8s} {'logit b→a':>18s} {'Δ':>8s} "
          f"{'%spoof b→a':>14s}")
    print("-" * 88)
    caught_new, lost = 0, 0
    for n in sorted(names):
        b, a = before[n], after[n]
        lb, la = b["mean_logit"], a["mean_logit"]
        sb, sa = b["frac_spoof_logit_gt0"] * 100, a["frac_spoof_logit_gt0"] * 100
        t = truth(n)
        flag = ""
        if t == "spoof":
            if lb < 0 <= la:
                flag = "  <== now CAUGHT"
                caught_new += 1
            elif la < 0 <= lb:
                flag = "  <== now MISSED"
                lost += 1
        elif t == "real" and la >= 0 > lb:
            flag = "  <== real->false-alarm!"
        print(f"{n:30s} {t:8s} {lb:8.2f} -> {la:7.2f} {la-lb:8.2f} "
              f"{sb:5.0f}% -> {sa:4.0f}%{flag}")
    print("-" * 88)

    spoofs = [n for n in names if truth(n) == "spoof"]
    b_caught = sum(1 for n in spoofs if before[n]["mean_logit"] >= 0)
    a_caught = sum(1 for n in spoofs if after[n]["mean_logit"] >= 0)
    print(f"spoof clips caught (mean logit>=0): before {b_caught}/{len(spoofs)}  "
          f"->  after {a_caught}/{len(spoofs)}   (newly caught {caught_new}, newly missed {lost})")
    reals = [n for n in names if truth(n) == "real"]
    for n in reals:
        verdict_b = "real" if before[n]["mean_logit"] < 0 else "FALSE-ALARM"
        verdict_a = "real" if after[n]["mean_logit"] < 0 else "FALSE-ALARM"
        print(f"real clip {n}: {verdict_b} -> {verdict_a} "
              f"(logit {before[n]['mean_logit']:.2f} -> {after[n]['mean_logit']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
