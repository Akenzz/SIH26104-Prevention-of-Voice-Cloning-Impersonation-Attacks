"""
data_pipeline/merge_hybrid_max_with_br.py
========================================
Stage 4: fold hybrid_br's training data into the hybrid_max corpus.

hybrid_br trained on `manifests/hybrid_plus_newclips_train.csv` (43,989 chunks on
`E:\\DatasetSIH\\processed_hybrid_vad\\`) = the shipped hybrid corpus + Amogh's
modern-engine clips. Keeping it is worth it for three things hybrid_max does not
have: ml/mr/ta (dropped from hybrid_max by instruction), 19 extra MLAAD engines,
and the `amogh_newclips` set whose recall win was measured through the real scipy
deployment path.

68% of it (29,912 rows) is ALREADY in hybrid_max train under the same
utterance_id, so this is a union-with-dedup, not a concat -- concatenating would
train on those clips twice.

THREE THINGS ARE FILTERED OUT, all for the same reason: they would contaminate
hybrid_max's measurement sets, which are kept verbatim.

  1. 137 rows whose utterance_id is in hybrid_max dev/eval/eval_ood.
     Straight train/test leakage.

  2. 531 rows across 12 generators that hybrid_max holds out for eval_ood
     (GPA-v1.5, Inworld-TTS-2, LongCat-AudioDiT, Maya1 TTS, Microsoft VibeVoice
     1.5B, NeuTTS-Nano, PrimeTTS, Qwen3-TTS-12Hz-0.6B-Base, optispeech,
     suno_bark, tts_models_en_ljspeech_fast_pitch, vixTTS). Training on these
     would make the headline unseen-generator number measure memorisation.

  3. 20 amogh_newclips rows whose informal generator name is the SAME ENGINE as a
     held-out one under a different spelling: chatterbox->Chatterbox,
     fireredtts->FireRedTTS-2.0, omni/qwen->Qwen2.5-Omni. This is the Edge-TTS
     bug in a different guise -- name-based hold-out does not catch aliases. 20
     rows of training data is worth less than 4 of the 25 held-out engines.

WHAT IS DELIBERATELY *NOT* FILTERED, and the caveat that comes with it
---------------------------------------------------------------------
14 train engines are a different SIZE or VERSION of a held-out engine's family
(Llasa-1B/3B vs held-out Llasa-8B, Higgs-Audio-V3 vs V2, Index-TTS-1.5 vs 2.0,
ElevenLabs-v3 vs Turbo-v2.5, Chatterbox-Turbo vs Chatterbox, ...). MLAAD treats
these as distinct generators and "unseen version of a seen family" is a fair --
arguably the most realistic -- generalisation test, so they stay in train.

  => Report eval_ood as "25 unseen generators, 14 of which have a same-family
     relative in training", never as 25 unseen architectures.

Shared vendor or dataset names are NOT family overlap and are ignored here:
`microsoft_speecht5_tts` shares only "Microsoft" with `Microsoft VibeVoice`, and
the eight `tts_models_en_ljspeech_*` Coqui recipes share a training DATASET, not
an engine.

Splits: hybrid_max's dev / eval / eval_ood pass through untouched. Only train
grows, and it is re-balanced to exact per-language 1:1 afterwards, because the
merged pools are not 1:1 on their own (br-train was itself chunk-imbalanced:
kn 486/1110, ta 231/580).

Deterministic: same SEED as the assembler.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from data_pipeline.assemble_hybrid_dataset import SEED  # noqa: E402
from data_pipeline.balance_hybrid_max_chunks import balance, report, verify  # noqa: E402

MANIFEST_DIR = repo_root / "data_pipeline" / "manifests"

# amogh_newclips generator names that are the same engine as a held-out generator
ALIAS_DROP = {"chatterbox", "fireredtts", "omni", "qwen"}

def _rd(p) -> pd.DataFrame:
    return pd.read_csv(p, encoding="utf-8-sig", dtype={"speaker_id": str},
                       low_memory=False)


def filter_br(br: pd.DataFrame, hold: pd.DataFrame) -> pd.DataFrame:
    """Drop every br-train row that would contaminate hybrid_max's eval sets."""
    ood_gen = set(hold[(hold.split == "eval_ood") & (hold.label == "spoof")]
                  .generator_id) - {"itw-unknown"}
    hold_utt = set(hold.utterance_id)
    hold_spk = set(hold[hold.label == "bonafide"].speaker_id)

    n0 = len(br)
    leak_utt = br.utterance_id.isin(hold_utt)
    leak_gen = br.generator_id.isin(ood_gen)
    leak_spk = (br.label == "bonafide") & br.speaker_id.isin(hold_spk)
    alias = (br.source_dataset == "amogh_newclips") & br.generator_id.isin(ALIAS_DROP)

    print(f"  br-train rows                    : {n0:,}")
    print(f"  - utterance_id in dev/eval/ood   : {int(leak_utt.sum()):,}")
    print(f"  - generator held out for eval_ood: {int(leak_gen.sum()):,} "
          f"({len(set(br.generator_id[leak_gen]))} generators)")
    print(f"  - speaker held out for dev/eval  : {int(leak_spk.sum()):,}")
    print(f"  - amogh alias of a held-out gen  : {int(alias.sum()):,} "
          f"({sorted(set(br.generator_id[alias]))})")
    out = br[~(leak_utt | leak_gen | leak_spk | alias)].copy()
    print(f"  = kept                           : {len(out):,}")
    return out


def family_overlap(train: pd.DataFrame, hold: pd.DataFrame) -> list[tuple[str, str]]:
    """Held-out generators that have a same-family relative in TRAIN (honesty check)."""
    import re

    def toks(s):
        s = re.sub(r"[^a-z0-9]+", " ", str(s).lower())
        # strip version/size/generic words so only the family name survives
        s = re.sub(r"\b(v?[0-9]+([._][0-9]+)*[a-z]?|tts|audio|model|models|dataset|"
                   r"multilingual|multi|base|large|nano|micro|mini|small|turbo|air|"
                   r"realtime|hz|ph|en|b)\b", " ", s)
        return {t for t in s.split() if len(t) > 3}

    # vendor / corpus names that are not engine families
    NOT_A_FAMILY = {"microsoft", "ljspeech", "coqui"}
    tr_gens = sorted(set(train[train.label == "spoof"].generator_id))
    ood = sorted(set(hold[(hold.split == "eval_ood") & (hold.label == "spoof")]
                     .generator_id) - {"itw-unknown"})
    pairs = []
    for o in ood:
        to = toks(o) - NOT_A_FAMILY
        for g in tr_gens:
            if g != o and (to & (toks(g) - NOT_A_FAMILY)):
                pairs.append((o, g))
                break
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Merge hybrid_br's train data into hybrid_max.")
    ap.add_argument("--max-manifest", default=str(MANIFEST_DIR / "hybrid_max_balanced.csv"))
    ap.add_argument("--br-manifest", default=str(MANIFEST_DIR / "hybrid_plus_newclips_train.csv"))
    ap.add_argument("--out", default=str(MANIFEST_DIR / "hybrid_maxbr.csv"))
    args = ap.parse_args()

    mx = _rd(args.max_manifest)
    br = _rd(args.br_manifest)
    mx_tr = mx[mx.split == "train"]
    hold = mx[mx.split != "train"]          # dev / eval / eval_ood, kept verbatim
    print(f"hybrid_max: {len(mx_tr):,} train + {len(hold):,} held-out chunks")

    print("\n----- filtering br-train against hybrid_max's eval sets -----")
    br_keep = filter_br(br, hold)

    print("\n----- union (dedup on utterance_id, hybrid_max wins) -----")
    merged = pd.concat([mx_tr, br_keep], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset="utterance_id", keep="first").reset_index(drop=True)
    print(f"  {before:,} -> {len(merged):,} rows (-{before-len(merged):,} duplicate chunks)")
    print(f"  net new training chunks from br : {len(merged)-len(mx_tr):,}")

    combined = pd.concat([merged, hold], ignore_index=True)
    report(combined, "MERGED, pre-balance")

    print("\n----- re-balancing to per-language 1:1 -----")
    rng = np.random.default_rng(SEED)
    out = balance(combined, rng)

    report(out, "FINAL (hybrid_max + hybrid_br data, chunk-level 1:1 per language)")

    print("\n----- verifying -----")
    ok = verify(combined, out)
    tr = out[out.split == "train"]
    fam = family_overlap(tr, out)
    print(f"\n  [NOTE] {len(fam)}/25 held-out generators have a same-family relative "
          f"in TRAIN -- quote eval_ood accordingly, not as unseen architectures:")
    for o, g in fam:
        print(f"         {o:34s} ~ train {g}")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False, encoding="utf-8-sig")
    for sp, g in out.groupby("split"):
        g.to_csv(outp.with_name(outp.stem + f"_{sp}.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[OK] {len(out):,} chunks -> {outp}")
    print("     per-split siblings: " +
          ", ".join(outp.stem + f"_{s}.csv" for s in sorted(out.split.unique())))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
