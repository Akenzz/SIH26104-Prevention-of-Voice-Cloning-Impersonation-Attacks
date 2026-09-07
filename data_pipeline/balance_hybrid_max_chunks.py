"""
data_pipeline/balance_hybrid_max_chunks.py
==========================================
Stage 3 for the hybrid_max corpus: restore exact 1:1 bonafide:spoof WITHIN each
language, at the CHUNK level.

Why this stage has to exist
---------------------------
assemble_hybrid_max_dataset.py balances base clips 1:1 per language, but the
trainer never sees base clips -- it reads the VAD chunk manifest, and chunk yield
is not constant across sources. Measured on this corpus (chunks per base clip):

    MLAAD-kn spoof     2.235      Kathbath-kn bonafide   1.008
    MLAAD-hi spoof     1.307      Gramvaani  bonafide    1.429
    MLAAD-en spoof     1.136      In-the-Wild bonafide   0.911

TTS engines emit longer, gap-free speech than the real recordings they are paired
with, and In-the-Wild additionally lost 2,410 bonafide clips to the <1 s drop
rule. Net effect on TRAIN before this stage:

    en  19,598 bona / 22,904 spoof   (1.17)
    hi  17,500 bona / 16,378 spoof   (0.94)
    kn     482 bona /  1,069 spoof   (2.22)  <-- 69% of Kannada chunks are spoof

A per-language class prior that steep is the "language ⇒ label" shortcut the 1:1
rule exists to prevent, so balancing globally is not enough -- it has to be per
language, which is why pipeline/balancer.py (global, plus a bonafide padding
hack) is not reused here.

eval_ood is deliberately left alone: recall is measured on its spoof side and
false-alarm on its bonafide side, independently, so it is not required to be 1:1
and its 25-generator spoof pool is worth more than symmetry.

Downsampling is stratified -- spoof by generator_id, bonafide by speaker_id --
so no generator and no speaker is dropped entirely.

Deterministic: same SEED as the assembler.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from data_pipeline.assemble_hybrid_max_dataset import LANG_NAME  # noqa: E402
from data_pipeline.assemble_hybrid_dataset import SEED, stratified_sample  # noqa: E402

BALANCE_SPLITS = ("train", "dev", "eval")   # eval_ood intentionally excluded
STRAT_KEY = {"spoof": "generator_id", "bonafide": "speaker_id"}

def balance(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Downsample the majority class to the minority count per (split, language)."""
    keep, dropped = [], []
    for split in sorted(df.split.unique()):
        sdf = df[df.split == split]
        if split not in BALANCE_SPLITS:
            keep.append(sdf)
            print(f"  {split:9s} untouched ({len(sdf):,} chunks)")
            continue
        for lang in sorted(sdf.language.unique()):
            ldf = sdf[sdf.language == lang]
            bo = ldf[ldf.label == "bonafide"]
            sp = ldf[ldf.label == "spoof"]
            n = min(len(bo), len(sp))
            if n == 0:                        # one-sided language: cannot pair it
                dropped.append(ldf)
                print(f"  {split:9s} {lang}: DROPPED entirely "
                      f"(bona={len(bo)} spoof={len(sp)}, cannot pair)")
                continue
            for lab, part in (("bonafide", bo), ("spoof", sp)):
                if len(part) == n:
                    keep.append(part)
                else:
                    cut = stratified_sample(part, n, STRAT_KEY[lab], rng)
                    keep.append(cut)
                    print(f"  {split:9s} {lang} {lab:8s}: {len(part):,} -> {n:,} "
                          f"(-{len(part)-n:,}, stratified by {STRAT_KEY[lab]})")
    out = pd.concat(keep, ignore_index=True)
    if dropped:
        print(f"  [WARN] {sum(len(d) for d in dropped):,} unpairable chunks removed")
    return out


def verify(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    """Re-assert every invariant the assembler guarantees, now at chunk level."""
    ok = True

    for split in BALANCE_SPLITS:
        s = after[after.split == split]
        for lang in sorted(s.language.unique()):
            l = s[s.language == lang]
            nb = int((l.label == "bonafide").sum())
            ns = int((l.label == "spoof").sum())
            if nb != ns:
                print(f"  [FAIL] {split} {lang} not 1:1 ({nb} vs {ns})"); ok = False

    # speakers must not span splits (chunks inherit the base row's split)
    spk = after[after.label == "bonafide"].groupby("speaker_id").split.nunique()
    if (spk > 1).any():
        bad = spk[spk > 1].index.tolist()[:5]
        print(f"  [FAIL] bonafide speakers in >1 split: {bad}"); ok = False

    # eval_ood generators must stay unseen by training
    tr_gen = set(after[(after.split == "train") & (after.label == "spoof")].generator_id)
    ood_gen = set(after[(after.split == "eval_ood") & (after.label == "spoof")].generator_id)
    leak = sorted(g for g in (tr_gen & ood_gen) if g != "itw-unknown")
    if leak:
        print(f"  [FAIL] eval_ood generators leaked into TRAIN: {leak}"); ok = False

    # no generator / speaker wiped out by the downsample
    for col, lab in (("generator_id", "spoof"), ("speaker_id", "bonafide")):
        b = set(before[(before.split == "train") & (before.label == lab)][col])
        a = set(after[(after.split == "train") & (after.label == lab)][col])
        if b - a:
            print(f"  [WARN] {len(b-a)}/{len(b)} train {lab} {col}s lost entirely: "
                  f"{sorted(b-a)[:5]}")

    print("  [OK] all invariants hold" if ok else "  [FAIL] invariants violated")
    return ok


def report(df: pd.DataFrame, title: str):
    print(f"\n===== {title} =====")
    piv = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    for c in ("bonafide", "spoof"):
        if c not in piv:
            piv[c] = 0
    print(piv[["bonafide", "spoof"]].to_string())
    print("\nTRAIN by language (the anti-shortcut invariant):")
    tr = df[df.split == "train"]
    for lang in sorted(tr.language.unique()):
        l = tr[tr.language == lang]
        nb = int((l.label == "bonafide").sum())
        ns = int((l.label == "spoof").sum())
        ratio = (ns / nb) if nb else float("inf")
        print(f"  {LANG_NAME.get(lang, lang):9s} {nb:7,} bona  {ns:7,} spoof   "
              f"ratio {ratio:.3f}")
    ngen = tr[tr.label == "spoof"].generator_id.nunique()
    print(f"\nspoof generators in TRAIN: {ngen}")


def main():
    ap = argparse.ArgumentParser(description="Chunk-level per-language 1:1 balancer.")
    ap.add_argument("--manifest", default=str(repo_root / "data_pipeline" / "manifests"
                                              / "hybrid_max_vad_chunks.csv"))
    ap.add_argument("--out", default=str(repo_root / "data_pipeline" / "manifests"
                                        / "hybrid_max_balanced.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, encoding="utf-8-sig", dtype={"speaker_id": str})
    report(df, "BEFORE (raw VAD chunks)")

    print("\n----- balancing -----")
    rng = np.random.default_rng(SEED)
    out = balance(df, rng)

    report(out, "AFTER (chunk-level 1:1 per language)")
    print("\n----- verifying -----")
    ok = verify(df, out)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False, encoding="utf-8-sig")
    for sp, g in out.groupby("split"):
        g.to_csv(outp.with_name(outp.stem + f"_{sp}.csv"), index=False,
                 encoding="utf-8-sig")
    print(f"\n[OK] {len(df):,} -> {len(out):,} chunks "
          f"(-{len(df)-len(out):,}) -> {outp}")
    print("     per-split siblings: " +
          ", ".join(outp.stem + f"_{s}.csv" for s in sorted(out.split.unique())))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
